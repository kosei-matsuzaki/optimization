"""ELSHADE-SPACMA — enhanced LSHADE-SPACMA (Hadi, Mohamed & Jambi, CEC 2018, 3rd place).

A. A. Hadi, A. W. Mohamed and K. M. Jambi, "Single-Objective Real-Parameter
Optimization: Enhanced LSHADE-SPACMA Algorithm", CEC 2018 competition entry
(technical report), later in "Heuristics for Optimization and Learning",
Springer, 2021, pp. 103-121.

Ported from the authors' MATLAB code as distributed by the competition
organisers (github.com/P-N-Suganthan/CEC2018, ``Bound-Constrained.rar`` ->
``Codes for best 3 (bound constrained).rar`` -> ``ELSHADE-SPACMA.zip``:
``ELSHADE_SPACMA.m``, ``LSHADE_SPACMA.m``, ``EADE_SPA.m``, ``genR_EADE.m``,
``CMAES_Init.m``, ``CMAES_update.m``, ``updateArchive.m``).

What is "enhanced" over LSHADE-SPACMA (``lshade_spacma.py``), all from the code:

1. Each outer iteration runs **two generations** on the same population:
   an LSHADE-SPACMA generation, then an **EADE** generation
   (``EADE_SPA.m``): ``v = x_mid + F (x_best - x_worst)``, where x_best is
   drawn from the top ceil(NP/10) ranks, x_worst from the bottom ceil(NP/10),
   x_mid from the rest (``genR_EADE.m``), F ~ U(0, 1) per individual, followed
   by binomial crossover with the SPA crossover rate. The F/CR memories (and
   the archive) are shared by both generations; the EADE generation updates
   the M_F memory with the SPA ``sf`` values it drew but did not use for the
   mutation (as in the code). It does not touch FCP or CMA-ES.
2. **Dynamic p** for current-to-pbest: p decreases linearly with nfes from
   0.3 to 0.15 (updated once per outer iteration, after LPSR).
3. LPSR (18·D -> 4) is applied once per outer iteration, i.e. after both
   generations; after the first reduction the archive capacity becomes NP
   (not 1.4·NP), as in the code.
4. Selection, archive and memory use ``u <= x`` (ties count as successes).

Everything else is the LSHADE-SPACMA generation of the same code base:
SPA (F ~ U[0.45, 0.55] in the first half, Cauchy(M_F, 0.1) in the second),
CR ~ N(M_CR, 0.1) with terminal value, FCP memory (L_Rate 0.8, clip
[0.2, 0.8]) choosing per individual between current-to-pbest/1/archive and a
CMA-ES sample, CMA-ES updated from the best floor(NP/2) of the population
after the LSHADE-SPACMA generation.

Parameters (default — source):
  pop_size_factor = 18     NP_init = 18·D               — ELSHADE_SPACMA.m
  min_pop_size    = 4                                   — ELSHADE_SPACMA.m
  memory_size     = 5      H                            — ELSHADE_SPACMA.m
  p_best_max/min  = 0.3 / 0.15 linear in nfes          — ELSHADE_SPACMA.m
  arc_rate        = 1.4    initial |A| = 1.4·NP, later NP — ELSHADE_SPACMA.m
  fcp_init        = 0.5    First_calss_percentage       — ELSHADE_SPACMA.m
  fcp_lr          = 0.8    L_Rate                       — LSHADE_SPACMA.m
  fcp_min/max     = 0.2 / 0.8                           — LSHADE_SPACMA.m
  f_spa_lo/width  = 0.45 / 0.1                          — LSHADE_SPACMA.m, EADE_SPA.m
  sigma0          = 0.5, xmean0 = rand(D) in [0,1]^D    — CMAES_Init.m
  eade_top_frac   = 0.1    T = ceil(NP/10)              — genR_EADE.m
CMA constants (cc, cs, c1, cmu, damps) computed once from the initial NP;
weights/mueff recomputed from the current NP at each update (CMAES_update.m).

Quirks reproduced from the code: NaN produced by a 0/0 memory update
(all successes are ties) propagates as in MATLAB, where ``min(NaN, 1) = 1``
(so the affected F / CR become 1 and the FCP slot becomes 0.8); a CMA update
that hits a non-finite C is discarded (MATLAB ``try/catch`` keeps the old
CMA state), and the authors' ``if flag==0, Hybridization_flag=1`` means it
does not switch hybridisation off. A negative eigenvalue gives MATLAB a
complex ``diagD``; the next LSHADE-SPACMA generation then sees a complex
mutant, switches hybridisation off for good and returns without evaluating.

Deviations: the budget is checked per evaluation (the last generation is
evaluated only up to ``max_evals``); the skipped generation of the
complex-mutant case consumes no evaluations here (MATLAB still adds NP to
nfes); RNG is NumPy's, seeded from ``self.seed`` (the code seeds from the
clock); bounds come from ``self.bounds`` instead of [-100, 100].
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


def _mround(v: float) -> float:
    """MATLAB round (half away from zero)."""
    return float(np.sign(v) * np.floor(np.abs(v) + 0.5))


def _cma_weights(pop_size: int):
    mu_f = pop_size / 2.0
    mu = int(np.floor(mu_f))
    w = np.log(mu_f + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = w.sum() ** 2 / np.sum(w ** 2)
    return mu, w, mueff


class ELSHADESPACMAOptimizer(BaseOptimizer):
    """ELSHADE-SPACMA (Hadi, Mohamed & Jambi, CEC 2018). See module docstring."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size_factor: int = 18,
        min_pop_size: int = 4,
        memory_size: int = 5,
        p_best_max: float = 0.3,
        p_best_min: float = 0.15,
        arc_rate: float = 1.4,
        fcp_init: float = 0.5,
        fcp_lr: float = 0.8,
        fcp_min: float = 0.2,
        fcp_max: float = 0.8,
        f_spa_lo: float = 0.45,
        f_spa_width: float = 0.1,
        sigma0: float = 0.5,
        eade_top_frac: float = 0.1,
    ):
        super().__init__(benchmark, seed)
        self.pop_size_factor = pop_size_factor
        self.min_pop_size = min_pop_size
        self.memory_size = memory_size
        self.p_best_max = p_best_max
        self.p_best_min = p_best_min
        self.arc_rate = arc_rate
        self.fcp_init = fcp_init
        self.fcp_lr = fcp_lr
        self.fcp_min = fcp_min
        self.fcp_max = fcp_max
        self.f_spa_lo = f_spa_lo
        self.f_spa_width = f_spa_width
        self.sigma0 = sigma0
        self.eade_top_frac = eade_top_frac

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        with np.errstate(all="ignore"):
            return self._optimize(max_evals)

    def _optimize(self, max_evals: int) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        D = self.dim
        H = self.memory_size

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        def evaluate(X: np.ndarray) -> np.ndarray:
            out = []
            for x in X:
                if len(history_f) >= max_evals:
                    break
                f = float(self.func(x))
                history_x.append(x.copy())
                history_f.append(f)
                out.append(f)
            return np.asarray(out, dtype=float)

        NP = int(self.pop_size_factor * D)
        max_NP = NP
        pop = lo + rng.random((NP, D)) * (hi - lo)
        fit = evaluate(pop)
        if len(fit) < NP:
            return self._make_result(history_x, history_f, [pop[:len(fit)].copy()])
        history_pop.append(pop.copy())

        st = {
            "p_best": self.p_best_max,
            "m_sf": np.full(H, 0.5),
            "m_cr": np.full(H, 0.5),
            "m_fcp": np.full(H, self.fcp_init),
            "pos": 0,
            "hybrid": True,
            "complex": False,  # MATLAB: diagD became complex at the last eig
        }
        archive = np.zeros((0, D))
        arc_cap = self.arc_rate * NP  # archive.NP (float; floor() on trimming)

        # --- CMA-ES state (CMAES_Init.m) ---
        _, w0, mueff0 = _cma_weights(NP)
        cma = {
            "sigma": self.sigma0,
            "xmean": rng.random(D),
            "cc": (4 + mueff0 / D) / (D + 4 + 2 * mueff0 / D),
            "cs": (mueff0 + 2) / (D + mueff0 + 5),
            "c1": 2 / ((D + 1.3) ** 2 + mueff0),
            "pc": np.zeros(D), "ps": np.zeros(D),
            "B": np.eye(D), "diagD": np.ones(D), "C": np.eye(D),
            "invsqrtC": np.eye(D), "eigeneval": 0,
            "chiN": D ** 0.5 * (1 - 1 / (4 * D) + 1 / (21 * D ** 2)),
        }
        cma["cmu"] = min(1 - cma["c1"],
                         2 * (mueff0 - 2 + 1 / mueff0) / ((D + 2) ** 2 + mueff0))
        cma["damps"] = 1 + 2 * max(0.0, np.sqrt((mueff0 - 1) / (D + 1)) - 1) + cma["cs"]

        def sample_cr_sf(n: int, nfes: int):
            idx = rng.integers(0, H, n)
            mu_sf, mu_cr = st["m_sf"][idx], st["m_cr"][idx]
            cr = rng.normal(mu_cr, 0.1)  # NaN mean -> NaN (as normrnd)
            cr[mu_cr == -1] = 0.0
            cr = np.fmax(np.fmin(cr, 1.0), 0.0)  # MATLAB min/max ignore NaN
            if nfes <= max_evals / 2:
                sf = self.f_spa_lo + self.f_spa_width * rng.random(n)
            else:
                sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(n) - 0.5))
                bad = sf <= 0
                while bad.any():
                    sf[bad] = mu_sf[bad] + 0.1 * np.tan(
                        np.pi * (rng.random(int(bad.sum())) - 0.5))
                    bad = sf <= 0
            sf = np.fmin(sf, 1.0)
            return idx, cr, sf

        def repair(X: np.ndarray, parent: np.ndarray) -> np.ndarray:
            low = X < lo
            X[low] = (parent[low] + lo) / 2
            high = X > hi
            X[high] = (parent[high] + hi) / 2
            return X

        def crossover(X: np.ndarray, parent: np.ndarray, cr: np.ndarray) -> np.ndarray:
            n = len(parent)
            mask = rng.random((n, D)) > cr[:, None]
            mask[np.arange(n), rng.integers(0, D, n)] = False
            return np.where(mask, parent, X)

        def update_archive(arc: np.ndarray, new: np.ndarray) -> np.ndarray:
            if arc_cap == 0 or len(new) == 0:
                return arc
            allp = np.vstack([arc, new]) if len(arc) else new.copy()
            uniq = np.unique(allp, axis=0)
            if len(uniq) < len(allp):
                allp = uniq
            if len(allp) > arc_cap:
                allp = allp[rng.permutation(len(allp))[:int(np.floor(arc_cap))]]
            return allp

        def update_memory(success, cr, sf, dif, cls1=None):
            good_cr, good_f = cr[success], sf[success]
            if len(good_f) == 0:
                return
            dv = dif[success]
            dv = dv / dv.sum()  # 0/0 -> NaN, as in MATLAB
            k = st["pos"]
            st["m_sf"][k] = np.dot(dv, good_f ** 2) / np.dot(dv, good_f)
            if good_cr.max() == 0 or st["m_cr"][k] == -1:
                st["m_cr"][k] = -1
            else:
                st["m_cr"][k] = np.dot(dv, good_cr ** 2) / np.dot(dv, good_cr)
            if cls1 is not None and st["hybrid"]:
                d1 = dif[success & cls1].sum()
                d2 = dif[success & ~cls1].sum()
                v = st["m_fcp"][k] * self.fcp_lr + (1 - self.fcp_lr) * d1 / (d1 + d2)
                st["m_fcp"][k] = np.fmax(np.fmin(v, self.fcp_max), self.fcp_min)
            st["pos"] = (k + 1) % H

        def cma_update(nfes: int):
            """CMAES_update.m; on failure keep the previous state (try/catch)."""
            n = len(pop)
            mu, w, mueff = _cma_weights(n)
            c = cma
            order = np.argsort(fit, kind="stable")[:mu]
            xold = c["xmean"]
            xmean = w @ pop[order]
            ps = (1 - c["cs"]) * c["ps"] + np.sqrt(c["cs"] * (2 - c["cs"]) * mueff) * (
                c["invsqrtC"] @ (xmean - xold)) / c["sigma"]
            hsig = (np.sum(ps ** 2) / (1 - (1 - c["cs"]) ** (2 * nfes / n)) / D
                    < 2 + 4 / (D + 1))
            pc = (1 - c["cc"]) * c["pc"] + hsig * np.sqrt(
                c["cc"] * (2 - c["cc"]) * mueff) * (xmean - xold) / c["sigma"]
            artmp = (pop[order] - xold) / c["sigma"]
            C = ((1 - c["c1"] - c["cmu"]) * c["C"]
                 + c["c1"] * (np.outer(pc, pc) + (1 - hsig) * c["cc"] * (2 - c["cc"]) * c["C"])
                 + c["cmu"] * (artmp.T * w) @ artmp)
            sigma = c["sigma"] * np.exp((c["cs"] / c["damps"])
                                        * (np.linalg.norm(ps) / c["chiN"] - 1))
            B, diagD, invsqrtC, eigeneval = c["B"], c["diagD"], c["invsqrtC"], c["eigeneval"]
            if nfes - eigeneval > n / (c["c1"] + c["cmu"]) / D / 10:
                eigeneval = nfes
                C = np.triu(C) + np.triu(C, 1).T
                if not np.all(np.isfinite(C)):
                    return  # eig() errors on NaN/Inf -> catch -> state unchanged
                evals, B = np.linalg.eigh(C)
                st["complex"] = bool(np.any(evals < 0))  # complex diagD in MATLAB
                diagD = np.sqrt(np.abs(evals))
                invsqrtC = (B / diagD) @ B.T
            c.update(xmean=xmean, ps=ps, pc=pc, C=C, sigma=sigma, B=B,
                     diagD=diagD, invsqrtC=invsqrtC, eigeneval=eigeneval)

        def finish_partial(trials: np.ndarray, child: np.ndarray) -> None:
            n_done = len(child)
            imp = child <= fit[:n_done]
            pop[:n_done][imp] = trials[:n_done][imp]
            fit[:n_done][imp] = child[imp]
            history_pop.append(pop.copy())

        while len(history_f) < max_evals:
            # ---------- LSHADE-SPACMA generation (LSHADE_SPACMA.m) ----------
            nfes = len(history_f)
            n = len(pop)
            sorted_index = np.argsort(fit, kind="stable")
            idx, cr, sf = sample_cr_sf(n, nfes)
            ratio = rng.random(n)
            cls1 = st["m_fcp"][idx] >= ratio
            if not st["hybrid"]:
                cls1[:] = True

            pop_all = np.vstack([pop, archive]) if len(archive) else pop
            r0 = np.arange(n)
            r1 = rng.integers(0, n, n)
            while (bad := r1 == r0).any():
                r1[bad] = rng.integers(0, n, int(bad.sum()))
            r2 = rng.integers(0, len(pop_all), n)
            while (bad := (r2 == r1) | (r2 == r0)).any():
                r2[bad] = rng.integers(0, len(pop_all), int(bad.sum()))
            pNP = max(int(_mround(st["p_best"] * n)), 2)
            pbest = pop[sorted_index[rng.integers(0, pNP, n)]]

            X = np.zeros((n, D))
            X[cls1] = pop[cls1] + sf[cls1, None] * (
                pbest[cls1] - pop[cls1] + pop[r1[cls1]] - pop_all[r2[cls1]])
            n2 = int((~cls1).sum())
            if n2:
                z = rng.standard_normal((n2, D))
                X[~cls1] = cma["xmean"] + cma["sigma"] * (z * cma["diagD"]) @ cma["B"].T

            if n2 and st["complex"]:
                # ~isreal(X): hybridisation off, generation returns unevaluated
                st["hybrid"] = False
            else:
                X = repair(X, pop)
                nan = np.isnan(X)
                X[nan] = pop[nan]
                ui = crossover(X, pop, cr)
                child = evaluate(ui)
                if len(child) < n:
                    finish_partial(ui, child)
                    break
                imp = child <= fit
                dif = np.abs(fit - child)
                archive = update_archive(archive, pop[imp])
                update_memory(imp, cr, sf, dif, cls1)
                pop[imp] = ui[imp]
                fit[imp] = child[imp]
                if st["hybrid"]:
                    cma_update(nfes)
            history_pop.append(pop.copy())
            if len(history_f) >= max_evals:
                break

            # ---------------- EADE generation (EADE_SPA.m) ----------------
            nfes = len(history_f)
            idx, cr, sf = sample_cr_sf(n, nfes)
            Fm = rng.random(n)
            order = np.argsort(fit, kind="stable")
            T = int(np.ceil(n * self.eade_top_frac))
            best_s, mid_s, worst_s = order[:T], order[T:n - T], order[n - T:]
            rb = best_s[rng.integers(0, len(best_s), n)]
            rw = worst_s[rng.integers(0, len(worst_s), n)]
            rm = mid_s[rng.integers(0, len(mid_s), n)]
            while (bad := rb == rw).any():
                rw[bad] = worst_s[rng.integers(0, len(worst_s), int(bad.sum()))]
            while (bad := rw == rm).any():
                rm[bad] = mid_s[rng.integers(0, len(mid_s), int(bad.sum()))]
            X = pop[rm] + Fm[:, None] * (pop[rb] - pop[rw])
            X = repair(X, pop)
            ui = crossover(X, pop, cr)
            child = evaluate(ui)
            if len(child) < n:
                finish_partial(ui, child)
                break
            imp = child <= fit
            dif = np.abs(fit - child)
            archive = update_archive(archive, pop[imp])
            update_memory(imp, cr, sf, dif, None)
            pop[imp] = ui[imp]
            fit[imp] = child[imp]

            # ------------- LPSR and dynamic p (ELSHADE_SPACMA.m) -------------
            nfes = len(history_f)
            plan = int(_mround((self.min_pop_size - max_NP) / max_evals * nfes + max_NP))
            st["p_best"] = ((self.p_best_min - self.p_best_max) / max_evals * nfes
                            + self.p_best_max)
            if n > plan:
                n_red = n - plan
                if n - n_red < self.min_pop_size:
                    n_red = n - self.min_pop_size
                if n_red > 0:
                    keep = np.sort(np.argsort(fit, kind="stable")[:n - n_red])
                    pop, fit = pop[keep], fit[keep]
                    arc_cap = len(pop)
                    if len(archive) > arc_cap:
                        archive = archive[rng.permutation(len(archive))[:arc_cap]]
            history_pop.append(pop.copy())

        return self._make_result(history_x, history_f, history_pop)
