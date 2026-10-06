"""HSES — Hybrid Sampling Evolution Strategy (Zhang & Shi, IEEE CEC 2018 winner).

G. Zhang and Y. Shi, "Hybrid Sampling Evolution Strategy for Solving Single
Objective Bound Constrained Problems", Proc. IEEE CEC 2018, pp. 1-7,
doi:10.1109/CEC.2018.8477908.

Three stages, each seeding the next:

1. Univariate sampling (a diagonal-Gaussian EDA): N(mean, std) per dimension,
   estimated from the weighted best ``mu`` of ``total`` samples, for ``I1``
   generations.
2. CMA-ES restarts (own implementation, as in the authors' code) started from
   the stage-1 best with a small step size, supplying the covariance
   information the univariate model cannot represent.
3. Univariate sampling again until the budget is spent, with the dimensions
   on which the CMA-ES restarts agreed frozen at the CMA-ES best.

Sources for the parameters
--------------------------
The paper itself was not readable here (IEEE / ResearchGate paywalls). The
staging is confirmed by Zhang et al., "Adaptive Structural Hyper-Parameter
Configuration by Q-Learning" (arXiv:2003.00863), Alg. 2 — UniSampling(I1) →
CMA-ES(θ) → Detect → UniSampling(I2) — which also states "In the original HSES,
the switch iteration is fixed to be 100". Every other constant below is taken
from the MATLAB port ``algorithm/hses.m`` in github.com/sefaaras/heuristic,
which states it was ported from the authors' own release (HSES.m, "Codes for
best 3" in Suganthan's CEC2018 repository). That code was written for the CEC
box [-100, 100]^D and a 10000·D budget; position-space constants are scaled by
``sc = mean(ub - lb) / 200`` (exact on a symmetric box), as in the port.

  stage 1: total = 200, mu = 100, I1 = 100 generations, log-rank weights;
           step shrinks to 0.96·randn once the best has not improved for 20
           generations (checked every 20 gens after gen 30; sticky)
  stage 2: lambda = floor(3 ln D) + 80, mu = lambda/2, sigma0 = 0.2·sc,
           Times = 2 restarts (D <= 30) else 1, each capped at maxFE/4 (D <= 30)
           else maxFE/2; a restart also stops as soon as the generation best
           changes by < 1e-11 from the previous generation best; both restarts
           start at the stage-1 best. Standard CMA-ES learning rates, with the
           authors' cmu denominator ((N+2)^2 + mueff) kept as written.
  stage 3: total/mu = 200/160 (D <= 30), 450/360 (D == 50), 600/480 otherwise
           (+200/+160 if D >= 50 and <= 30% budget used); dims frozen at the
           CMA-ES best with std 0.001·sc for the first generation.
  bounds:  stage-1 loop and CMA-ES use the modulo wrap about the box centre;
           sampled stage-1-init/stage-3 points that violate the box are
           resampled (10 tries) then clamped; every evaluated point is clamped.

Budget note
-----------
Stage 1 is a fixed 200·(I1+1) = 20 200 evaluations, which is ~20% of the CEC
10-D budget (100 000) but more than the whole quick budget at 2-D (5 000). With
the defaults the method is faithful to the competition code; on small budgets
it degenerates to stage 1 only. ``stage1_frac`` (not in the paper) rescales I1
so stage 1 takes that fraction of ``max_evals`` — use e.g. 0.2 to mimic the
10-D CEC ratio. This is our guess, not the authors'.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    pass


class HSESOptimizer(BaseOptimizer):
    """Hybrid Sampling Evolution Strategy (CEC 2018 winner), port of the authors' code."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        stage1_total: int = 200,          # authors' code
        stage1_mu: int = 100,             # authors' code
        stage1_iters: int = 100,          # I1; "fixed to be 100" (arXiv:2003.00863)
        stage1_frac: Optional[float] = None,  # OUR option: I1 from a budget fraction
        cma_sigma0: float = 0.2,          # authors' code, on the [-100,100] scale
        cma_lambda_extra: int = 80,       # lambda = floor(3 ln D) + 80, authors' code
    ):
        super().__init__(benchmark, seed)
        self.stage1_total = stage1_total
        self.stage1_mu = stage1_mu
        self.stage1_iters = stage1_iters
        self.stage1_frac = stage1_frac
        self.cma_sigma0 = cma_sigma0
        self.cma_lambda_extra = cma_lambda_extra

    # ------------------------------------------------------------------ helpers
    def _eval(self, X: np.ndarray) -> np.ndarray:
        """Clamp and evaluate a block of rows; truncate at the budget."""
        X = np.clip(np.atleast_2d(X), self._lo, self._hi)
        remaining = self._max_evals - len(self._hf)
        if remaining <= 0:
            raise _BudgetExhausted()
        n = min(len(X), remaining)
        fit = np.empty(n)
        for i in range(n):
            f = float(self.func(X[i]))
            fit[i] = f
            self._hx.append(X[i].copy())
            self._hf.append(f)
            if f < self._bsf:
                self._bsf = f
                self._bsfx = X[i].copy()
        self._hp.append(X[:n].copy())
        if n < len(X):
            raise _BudgetExhausted()
        return fit

    def _wrap(self, X: np.ndarray) -> np.ndarray:
        """Authors' mod(x, ±100) repair, generalised about the box centre."""
        X = X.copy()
        c, h = self._ctr, self._hw
        hi_v = X > self._hi
        if hi_v.any():
            # np.mod follows the divisor's sign, like MATLAB mod: [0, h)
            X[hi_v] = c + np.mod(X[hi_v] - c, h)
        lo_v = X < self._lo
        if lo_v.any():
            X[lo_v] = c + np.mod(X[lo_v] - c, -h)          # (-h, 0]
        return X

    def _resample_out(self, X, meanval, stdval, rng):
        X = X.copy()
        M = np.broadcast_to(meanval, X.shape)
        S = np.broadcast_to(stdval, X.shape)
        for _ in range(10):
            bad = (X < self._lo) | (X > self._hi)
            if not bad.any():
                return X
            X[bad] = M[bad] + S[bad] * rng.standard_normal(int(bad.sum()))
        return np.clip(X, self._lo, self._hi)

    @staticmethod
    def _weights(mu: int) -> np.ndarray:
        w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        return w / w.sum()

    # ----------------------------------------------------------------- optimize
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        D = self.dim
        self._lo, self._hi = float(lo), float(hi)
        self._ctr = (self._lo + self._hi) / 2
        self._hw = (self._hi - self._lo) / 2
        sc = (self._hi - self._lo) / 200.0
        self._max_evals = max_evals
        self._hx, self._hf, self._hp = [], [], []
        self._bsf = math.inf
        self._bsfx = rng.uniform(lo, hi, D)

        I1 = self.stage1_iters
        if self.stage1_frac is not None:
            I1 = max(1, int(self.stage1_frac * max_evals / self.stage1_total) - 1)

        try:
            self._run(rng, D, sc, I1, max_evals)
        except _BudgetExhausted:
            pass
        return self._make_result(self._hx, self._hf, self._hp)

    def _run(self, rng, D, sc, I1, max_evals):
        lo, hi = self._lo, self._hi
        # ---------------- Stage 1: univariate Gaussian sampling
        total, mu = self.stage1_total, self.stage1_mu
        pos = rng.uniform(lo, hi, (total, D))
        e = self._eval(pos)
        weights = self._weights(mu)
        top = pos[np.argsort(e, kind="stable")[:mu]]
        meanval = top.mean(axis=0)
        stdval = top.std(axis=0, ddof=1)
        pos = meanval + stdval * rng.standard_normal((total, D))
        pos = self._resample_out(pos, meanval, stdval, rng)

        cc1 = False
        FV = np.zeros(max(I1, 1))
        a1_first = float(np.min(e))
        for kk in range(1, I1 + 1):
            e = self._eval(pos)
            order = np.argsort(e, kind="stable")
            a1_first = float(e[order[0]])
            newpos = pos[order[:mu]]
            meanval = weights @ newpos
            stdval = newpos.std(axis=0, ddof=1)
            FV[kk - 1] = a1_first
            if kk > 30 and kk % 20 == 0:
                aa2 = int(np.argmin(FV[:kk])) + 1
                if aa2 < kk - 20:
                    cc1 = True
            step = (0.96 if cc1 else 1.0) * rng.standard_normal((total, D))
            pos = self._wrap(meanval + stdval * step)

        previousbest = a1_first
        bestvec = self._bsfx.copy()

        # ---------------- Stage 2: CMA-ES restarts from the stage-1 best
        if D <= 30:
            times, stopeval = 2, max_evals / 4
        else:
            times, stopeval = 1, max_evals / 2
        arfitnessbest = np.full(times, self._bsf)
        xvalbest = np.tile(bestvec[:, None], (1, times))

        N = D
        lam = int(math.floor(3 * math.log(N))) + self.cma_lambda_extra
        mucma = lam // 2
        wcma = math.log(lam / 2 + 0.5) - np.log(np.arange(1, mucma + 1))
        wcma = wcma / wcma.sum()
        mueff = wcma.sum() ** 2 / (wcma ** 2).sum()
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2)
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (N + 1)) - 1) + cs
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        for k in range(times):
            sigma = self.cma_sigma0 * sc
            pc = np.zeros(N)
            ps = np.zeros(N)
            B = np.eye(N)
            DD = np.ones(N)          # diagonal of D (sqrt eigenvalues)
            C = np.eye(N)
            eigenval = 0
            counteval = 0
            xmean = bestvec.copy()
            while counteval < stopeval:
                arz = rng.standard_normal((N, lam))
                arxx = xmean[:, None] + sigma * (B @ (DD[:, None] * arz))
                arxx = self._wrap(arxx.T).T
                arfit = self._eval(arxx.T)
                counteval += lam
                idx = np.argsort(arfit, kind="stable")
                arfit = arfit[idx]
                if abs(arfit[0] - previousbest) < 1e-11:
                    break
                previousbest = arfit[0]
                if arfitnessbest[k] > arfit[0]:
                    arfitnessbest[k] = arfit[0]
                    xvalbest[:, k] = arxx[:, idx[0]]
                sel = idx[:mucma]
                xmean = arxx[:, sel] @ wcma
                zmean = arz[:, sel] @ wcma
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
                hsig = (np.linalg.norm(ps)
                        / math.sqrt(1 - (1 - cs) ** (2 * counteval / lam))
                        / chiN) < 1.4 + 2 / (N + 1)
                pc = ((1 - cc) * pc
                      + hsig * math.sqrt(cc * (2 - cc) * mueff) * (B @ (DD * zmean)))
                BDz = B @ (DD[:, None] * arz[:, sel])
                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                     + cmu * (BDz * wcma) @ BDz.T)
                sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                if counteval - eigenval > lam / cmu / N / 10:
                    eigenval = counteval
                    C = np.triu(C) + np.triu(C, 1).T
                    ev, B = np.linalg.eigh(C)
                    DD = np.sqrt(np.maximum(ev, 0.0))
                if arfit[0] == arfit[int(math.ceil(0.7 * lam)) - 1]:
                    sigma = sigma * math.exp(0.2 + cs / damps)

        # ---------------- Stage 3: univariate sampling with frozen dimensions
        if D <= 30:
            total, mu = 200, 160
        elif D == 50:
            total, mu = 450, 360
        else:
            total, mu = 600, 480
        if D >= 50 and len(self._hf) <= 0.3 * max_evals:
            total += 200
            mu += 160
        weights = self._weights(mu)

        ppp1 = np.zeros(D)
        dividevalue = 0.0
        bbpbb = np.ones(D)
        if D <= 30:
            # freeze the dims on which the two CMA-ES restarts agreed
            ppp1 = xvalbest.std(axis=1, ddof=1)
            ppp2 = np.sort(ppp1)
            if ppp2[0] > 0.2 * sc:
                dividevalue = 0.0
            elif ppp2.max() < 0.01 * sc:
                dividevalue = 1.0 * sc
            else:
                ind = np.zeros(D)
                for dd in range(1, D):
                    ind[dd] = ((ppp2[dd] - ppp2[dd - 1]) / ppp2[dd - 1]
                               if ppp2[dd - 1] != 0 else math.inf)
                fin = ind[np.isfinite(ind)]
                ind[0] = (fin.min() if fin.size else 0.0) - 0.001
                value2 = np.argsort(-ind, kind="stable")
                for dd in range(D):
                    v = value2[dd]
                    if ppp2[v] < 10 * sc:
                        if ppp2[v] > 0.1 * sc:
                            dividevalue = ppp2[v] - 0.001 * sc
                            break
                    elif ppp2[max(v - 1, 0)] < 0.01 * sc:
                        dividevalue = ppp2[v] - 0.001 * sc
                        break
                    if dd == D - 1:
                        dividevalue = ppp2[v] - 0.001 * sc
        else:
            # coordinate-wise sensitivity probe around the CMA-ES best
            nprobe = int(round(total / 5))
            bbpbbp = np.zeros(D)
            base = xvalbest[:, 0].copy()
            spos = np.tile(base, (nprobe, 1))
            denom = max(abs(arfitnessbest[0]), np.finfo(float).eps)
            offs = (np.arange(1, nprobe + 1) - 0.1 * total) * (hi - lo) / 200
            for d in range(D):
                spos[:, d] = base[d] + offs
                ep = self._eval(spos)
                bbpbbp[d] = abs(ep.max() / denom)
                spos[:, d] = base[d]
            if bbpbbp.max() >= 3.1:
                aaa1 = np.sort(bbpbbp)
                di = np.array([aaa1[d + 1] / aaa1[d] if aaa1[d] != 0 else math.inf
                               for d in range(D - 1)])
                aab2 = np.argsort(-di, kind="stable")
                division = 0.0
                thr = 1.8 if aaa1[max(int(math.floor(D / 2 + 0.5)), 1) - 1] <= 2 else 4.0
                for d in range(D - 1):
                    if aaa1[aab2[d]] < thr:
                        division = aaa1[aab2[d]] + 0.01
                        break
                bbpbb = (bbpbbp <= division).astype(float)

        seq = int(np.argmin(arfitnessbest))
        xfrozen = xvalbest[:, seq].copy()
        frozen = (bbpbb == 0) if D > 30 else (ppp1 < dividevalue)

        pos = rng.uniform(lo, hi, (total, D))
        kk = 1
        cc2 = False
        xmin: list[float] = []
        while True:
            e1 = self._eval(pos)
            order = np.argsort(e1, kind="stable")
            xmin.append(float(e1[order[0]]))
            newpos = pos[order[:min(mu, total)]]
            meanval = weights[:len(newpos)] @ newpos
            stdval = newpos.std(axis=0, ddof=1)
            if kk == 1:
                stdval[frozen] = 0.001 * sc
                meanval[frozen] = xfrozen[frozen]
            kk += 1
            if kk > 30 and kk % 20 == 0:
                bbb = int(np.argmin(xmin)) + 1
                cc2 = bbb < kk - 20
            step = (0.96 if cc2 else 1.0) * rng.standard_normal((total, D))
            pos = self._resample_out(meanval + stdval * step, meanval, stdval, rng)
