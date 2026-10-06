"""EBOwithCMAR — Effective Butterfly Optimizer with Covariance Matrix Adapted
Retreat phase (Kumar, Misra & Singh, IEEE CEC 2017; 1st place, CEC 2017
bound-constrained single-objective competition).

This is a line-by-line port of the authors' MATLAB code as distributed by the
competition organisers (P-N-Suganthan/CEC2017-BoundContrained,
``Codes-of-Top-Methods-and-results.zip`` → ``EBOwithCMAR.zip``: files
``EBO_BIN.m`` (main loop), ``EBO.m`` (perching/patrolling = DE-like butterfly
operators), ``Scout.m`` (CMAR = CMA-ES retreat phase), ``LS2.m`` (SQP local
search), ``init_cma_par.m``, ``han_boun.m``, ``bestt.m``, ``gnR1R2.m``,
``updateArchive.m``, ``Introd_Par.m``). Where the code and the paper differ,
the code is followed — it is what produced the competition results.

Structure (per iteration):
  * Two sub-populations: EA_1 (size PS1 = 18·n, linear reduction to 4) evolved
    by EBO's two self-adaptive butterfly operators ("crisscross" and
    "towards-best" modifications), with success-history memories for F, CR,
    T (exponential crossover profile) and freq (sinusoidal F, first half of
    the budget), an external archive (2.6·PS1) and adaptive operator
    probabilities; EA_2 (PS2 = 4 + ⌊3 ln n⌋) evolved by the CMAR phase
    (CMA-ES with non-Gaussian arcsine-based sampling and fitness-proportional
    recombination weights).
  * Every CS iterations the two phases are scored by normalised quality +
    diversity; for the next CS iterations each phase runs with that
    probability, then information is shared (EA_1 → EA_2 with CMA re-init,
    or best of EA_2 → worst of EA_1).
  * After 75 % of the budget, with probability p_ls (0.1, dropping to 0.01
    after a failure) an SQP local search (MATLAB ``fmincon(...,'sqp')``,
    budget 2 % of max FEs) is started from the best-so-far.

Deviations / guesses (all others follow the code literally):
  1. Bound handling at evaluation: the code evaluates CMAR samples *outside*
     the box during the first 50 % of the run (CEC2017 functions are defined
     everywhere). Here every point is repaired into the box (the code's own
     ``han_boun`` type 2) before evaluation, because this repo scores the
     minimum of every recorded evaluation and an out-of-box point must not
     count. The CMA update still uses the unrepaired samples, as in the code.
  2. SQP local search: ``scipy.optimize.minimize(method="SLSQP")`` with box
     bounds replaces ``fmincon`` SQP (same budget: ceil(0.02·max_evals)
     evaluations, forward finite-difference gradients). The best point seen
     during the local search is returned (fmincon returns its last iterate).
  3. Per-dimension constants: ``Introd_Par.m`` only defines CS / G_max for
     n = 10, 30, 50, 100 and falls to the 100-D values otherwise; here
     n ≤ 10 uses the 10-D values (CS = 50, G_max = 2163), n ≤ 30 the 30-D
     ones, etc. G_max (the generation horizon of the sinusoidal-F schedule)
     was tuned for Max_FES = 10000·n, so it is scaled by max_evals/(10000·n).
  4. The early-stop at |f − f*| ≤ 1e-8 is dropped (f* is not given to
     optimizers in this repo); the run always uses the full budget.
  5. Numerical guards absent from the code: division by |f| in the operator
     credit uses |f| + 1e-300; if the CMA weights' sum is ≤ 0 / non-finite or
     the covariance becomes non-finite, uniform weights / a CMA re-init from
     EA_2 is used. MATLAB's NaN-ignoring min/max are reproduced with fmin/fmax.

Faithfully reproduced oddities (flagged so nobody "fixes" them silently):
  * CMAR recombination weights are the raw sorted fitness values of the μ best
    samples, normalised (``fliplr`` on a column vector is a no-op), so the
    *worse* of the μ best get the larger weight. On CEC2017 (f ≥ 100 bias)
    this is ≈ uniform; on f* = 0 suites it is not. ``cma_weights="uniform"``
    gives the ≈-uniform CEC behaviour instead (option, not default).
  * ``han_boun`` type 2 maps upper-bound violations with ``2·lo − x_old``,
    i.e. effectively to the lower bound.
  * Sinusoidal F uses ``tan`` (not ``sin`` as in LSHADE-cnEpSin), so the
    first branch gives F ≈ 0.5 and the second can give large or negative F.
  * The memory index advances between the CR and the T/freq updates.
  * CMA σ0 = 0.3 and σ = 1e-5 after a successful LS are absolute values
    (CEC box is [-100, 100]); kept as-is (``cma_sigma0``).
"""
from __future__ import annotations

import math

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    pass


def _matlab_round(v: float) -> int:
    return int(math.floor(v + 0.5))


class EBOwithCMAROptimizer(BaseOptimizer):
    """EBOwithCMAR (Kumar, Misra & Singh, CEC 2017 winner). Port of the
    authors' MATLAB code; see module docstring for deviations."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size1: int | None = None,   # code: PS1 = 18·n (Introd_Par.m)
        min_pop_size: int = 4,          # code: Par.MinPopSize
        arch_rate: float = 2.6,         # code: arch_rate (EBO_BIN.m)
        memory_size: int = 6,           # code: memory_size
        cycle: int | None = None,       # code: Par.CS (50 for n=10)
        g_max: float | None = None,     # code: Par.Gmax (2163 for n=10), scaled by budget
        prob_ls: float = 0.1,           # code: Par.prob_ls
        ls_budget_frac: float = 0.02,   # code: LS2.m Par.LS_FE = ceil(0.02·Max_FES)
        cma_sigma0: float = 0.3,        # code: init_cma_par.m insigma (absolute)
        cma_weights: str = "code",      # "code" (fitness-proportional) | "uniform"
    ):
        super().__init__(benchmark, seed)
        n = self.dim
        self.pop_size1 = 18 * n if pop_size1 is None else pop_size1
        self.min_pop_size = min_pop_size
        self.arch_rate = arch_rate
        self.memory_size = memory_size
        if n <= 10:
            cs_d, gmax_d = 50, 2163
        elif n <= 30:
            cs_d, gmax_d = 100, 2745
        elif n <= 50:
            cs_d, gmax_d = 150, 3022
        else:
            cs_d, gmax_d = 150, 3401
        self.cycle = cs_d if cycle is None else cycle
        self._gmax_default = gmax_d
        self.g_max = g_max
        self.prob_ls = prob_ls
        self.ls_budget_frac = ls_budget_frac
        self.cma_sigma0 = cma_sigma0
        if cma_weights not in ("code", "uniform"):
            raise ValueError("cma_weights must be 'code' or 'uniform'")
        self.cma_weights = cma_weights

    # ------------------------------------------------------------------ utils
    def _eval(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for k, x in enumerate(X):
            if len(self._hf) >= self._max_evals:
                raise _BudgetExhausted()
            x = np.asarray(x, dtype=float).copy()
            f = float(self.func(x))
            self._hx.append(x)
            self._hf.append(f)
            out[k] = f
        return out

    def _han_boun(self, x: np.ndarray, x2: np.ndarray, hb: int) -> np.ndarray:
        """han_boun.m (UMOEA-II). x, x2: (P, n)."""
        x = x.copy()
        L = np.broadcast_to(self._lo, x.shape)
        U = np.broadcast_to(self._hi, x.shape)
        x2 = np.broadcast_to(x2, x.shape)
        if hb == 1:  # DE: midpoint between parent and violated bound
            pos = x < L
            x[pos] = (x2[pos] + L[pos]) / 2
            pos = x > U
            x[pos] = (x2[pos] + U[pos]) / 2
        else:  # CMA-ES (upper branch uses 2·L − x2 exactly as in the code)
            pos = x < L
            x[pos] = np.fmin(U[pos], np.fmax(L[pos], 2 * L[pos] - x2[pos]))
            pos = x > U
            x[pos] = np.fmax(L[pos], np.fmin(U[pos], 2 * L[pos] - x2[pos]))
        return x

    def _init_cma(self, EA_2: np.ndarray, n: int, n2: int) -> dict:
        """init_cma_par.m"""
        s: dict = {}
        s["xmean"] = EA_2.mean(axis=0)
        s["sigma"] = float(self.cma_sigma0)
        s["pc"] = np.zeros(n)
        s["ps"] = np.zeros(n)
        s["B"] = np.eye(n)
        s["diagD"] = np.ones(n)
        s["BD"] = np.eye(n)
        s["C"] = np.eye(n)
        s["chiN"] = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        mu = int(math.ceil(n2 / 2))
        s["mu"] = mu
        w = np.log(max(mu, n / 2) + 0.5) - np.log(np.arange(1, mu + 1))
        mueff = w.sum() ** 2 / (w ** 2).sum()
        s["mueff"] = mueff
        s["weights"] = w / w.sum()
        s["cc"] = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        s["cs"] = (mueff + 2) / (n + mueff + 3)
        s["ccov1"] = 2 / ((n + 1.3) ** 2 + mueff)
        s["ccovmu"] = 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)
        s["damps"] = (0.5 + 0.5 * min(1.0, (0.27 * n2 / mueff - 1) ** 2)
                      + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + s["cs"])
        s["xold"] = s["xmean"].copy()
        return s

    @staticmethod
    def _gnR1R2(rng, NP1: int, NP2: int):
        """gnR1R2.m: r1∈[0,NP1)≠i, r2∈[0,NP2)≠r1,i, r3∈[0,NP1)≠i,r1,r2."""
        r0 = np.arange(NP1)
        r1 = rng.integers(0, NP1, NP1)
        while True:
            pos = r1 == r0
            if not pos.any():
                break
            r1[pos] = rng.integers(0, NP1, pos.sum())
        r2 = rng.integers(0, NP2, NP1)
        while True:
            pos = (r2 == r1) | (r2 == r0)
            if not pos.any():
                break
            r2[pos] = rng.integers(0, NP2, pos.sum())
        r3 = rng.integers(0, NP1, NP1)
        while True:
            pos = (r3 == r0) | (r3 == r1) | (r3 == r2)
            if not pos.any():
                break
            r3[pos] = rng.integers(0, NP1, pos.sum())
        return r1, r2, r3

    @staticmethod
    def _bestt(rng, P: int, D: int) -> np.ndarray:
        """bestt.m: index of the 'best' guide for towards-best."""
        k = P
        if 2 * D > P:
            D = 1
            P = max(_matlab_round(0.1 * P), 2)
        return np.array([rng.permutation(P)[:D].min() for _ in range(k)])

    # ---------------------------------------------------------------- phases
    def _ebo(self, rng, st: dict, gg: int) -> None:
        """EBO.m — perching / patrolling butterfly operators on EA_1."""
        x, fitx = st["EA_1"], st["obj1"]
        P, n = x.shape
        M = self.memory_size
        mem = rng.integers(0, M, P)
        mu_sf = st["af"][mem]
        mu_cr = st["acr"][mem]
        mu_T = st["aT"][mem]
        mu_freq = st["afreq"][mem]

        cr = mu_cr + 0.1 * math.sqrt(math.pi) * (np.arcsin(-rng.random(P)) + np.arcsin(rng.random(P)))
        cr[mu_cr == -1] = 0
        cr = np.fmax(np.fmin(cr, 1), 0)

        F = mu_sf + 0.1 * np.tan(math.pi * (rng.random(P) - 0.5))
        pos = np.where(F <= 0)[0]
        while pos.size:
            F[pos] = mu_sf[pos] + 0.1 * np.tan(math.pi * (rng.random(pos.size) - 0.5))
            pos = np.where(F <= 0)[0]
        F = np.fmin(F, 1)

        T = mu_T + 0.05 * (math.sqrt(math.pi) * (np.arcsin(-rng.random(P)) + np.arcsin(rng.random(P))))
        T = np.fmin(np.fmax(T, 0), 0.5)
        l = np.floor(n * rng.random(P)).astype(int)
        if n == 1:
            CR = cr[:, None].copy()
        else:
            CR = np.zeros((P, n))
            half = n // 2
            for i in range(P):
                if n % 2 == 0:
                    mm = np.exp(-T[i] / n * np.arange(half))
                    ll = cr[i] * np.concatenate([mm, mm[::-1]])
                else:
                    mm = np.exp(-T[i] / n * np.arange(half))
                    mm1 = math.exp(-T[i] / n * half)
                    ll = cr[i] * np.concatenate([mm, [mm1], mm[::-1]])
                cols = np.concatenate([np.arange(l[i], n), np.arange(0, l[i])])
                CR[i, cols] = ll

        freq = mu_freq + 0.1 * np.tan(math.pi * (rng.random(P) - 0.5))
        pos = np.where(freq <= 0)[0]
        while pos.size:
            freq[pos] = mu_freq[pos] + 0.1 * np.tan(math.pi * (rng.random(pos.size) - 0.5))
            pos = np.where(freq <= 0)[0]
        freq = np.fmin(freq, 1)
        G_Max = st["Gmax"]
        if len(self._hf) <= self._max_evals / 2:
            if rng.random() < 0.5:
                F = 0.5 * (math.tan(2 * math.pi * 0.5 * gg + math.pi) * ((G_Max - gg) / G_Max) + 1) * np.ones(P)
            else:
                F = 0.5 * (np.tan(2 * math.pi * freq * gg) * (gg / G_Max) + 1)

        archive = st["archive"]
        popAll = np.vstack([x, archive]) if len(archive) else x
        r1, r2, r3 = self._gnR1R2(rng, P, len(popAll))

        bb = rng.random(P)
        prob = st["probDE1"]
        l2 = prob[0] + prob[1]
        op_1 = bb <= prob[0]
        op_2 = (bb > prob[0]) & (bb <= l2)

        randindex = self._bestt(rng, P, n)
        phix = x[randindex]

        vi = np.zeros((P, n))
        Fc = F[:, None]
        vi[op_1] = x[op_1] + Fc[op_1] * (x[r1[op_1]] - x[op_1] + x[r3[op_1]] - popAll[r2[op_1]])
        vi[op_2] = x[op_2] + Fc[op_2] * (phix[op_2] - x[op_2] + x[r1[op_2]] - x[r3[op_2]])
        vi = self._han_boun(vi, x, 1)

        mask = rng.random((P, n)) > CR
        cols = np.floor(rng.random(P) * n).astype(int)
        mask[np.arange(P), cols] = False
        ui = vi.copy()
        ui[mask] = x[mask]

        fitx_new = self._eval(ui)  # may raise; whole batch evaluated below otherwise

        diff = np.abs(fitx - fitx_new)
        I = fitx_new < fitx
        goodCR, goodF, goodT, goodFreq = cr[I], F[I], T[I], freq[I]

        # archive (updateArchive.m)
        if I.any() and st["archNP"] > 0:
            A = np.vstack([archive, x[I]]) if len(archive) else x[I].copy()
            _, ix = np.unique(A, axis=0, return_index=True)
            A = A[np.sort(ix)] if len(ix) < len(A) else A
            NP = int(st["archNP"])  # MATLAB indexes 1:archive.NP with non-integer NP → floor
            if len(A) > NP:
                A = A[rng.permutation(len(A))[:NP]]
            st["archive"] = A

        diff2 = np.maximum(0, fitx - fitx_new) / (np.abs(fitx) + 1e-300)
        count_S = np.zeros(2)
        for k, op in enumerate((op_1, op_2)):
            m = diff2[op].mean() if op.any() else np.nan
            count_S[k] = np.fmax(0.0, m)
        if np.all(count_S != 0):
            st["probDE1"] = np.fmax(0.1, np.fmin(0.9, count_S / count_S.sum()))
        else:
            st["probDE1"] = 0.5 * np.ones(2)

        fitx = fitx.copy()
        x = x.copy()
        fitx[I] = fitx_new[I]
        x[I] = ui[I]

        if goodCR.size > 0:
            # 0/0 → NaN exactly as in MATLAB; NaN memories are later neutralised
            # by the NaN-ignoring fmin/fmax clamps (MATLAB min/max semantics).
            _err = np.errstate(invalid="ignore", divide="ignore")
            _err.__enter__()
            w = diff[I] / diff[I].sum()
            h = st["hist_pos"]
            st["af"][h] = (w @ goodF ** 2) / (w @ goodF)
            if goodCR.max() == 0 or st["acr"][h] == -1:
                st["acr"][h] = -1
            else:
                st["acr"][h] = (w @ goodCR ** 2) / (w @ goodCR)
            h += 1
            if h >= M:
                h = 0
            st["hist_pos"] = h
            st["aT"][h] = (w @ goodT ** 2) / (w @ goodT)
            if goodFreq.max() == 0 or st["afreq"][h] == -1:
                st["afreq"][h] = -1
            else:
                st["afreq"][h] = (w @ goodFreq ** 2) / (w @ goodFreq)
            _err.__exit__(None, None, None)

        order = np.argsort(fitx, kind="stable")
        st["EA_1"], st["obj1"] = x[order], fitx[order]

    def _scout(self, rng, st: dict, it: int) -> None:
        """Scout.m — CMAR (CMA-ES retreat) phase on EA_2."""
        s = st["cma"]
        x_old = st["EA_2"]
        n = self.dim
        P = x_old.shape[0]
        if rng.random() < st["probSC"][0]:
            arz = math.sqrt(math.pi) * (np.arcsin(rng.random((n, P))) + np.arcsin(-rng.random((n, P))))
        else:
            arz = math.sqrt(math.pi) * np.arcsin(2 * rng.random((n, P)) - 1)
        arx = s["xmean"][:, None] + s["sigma"] * (s["BD"] @ arz)
        second_half = len(self._hf) >= 0.5 * self._max_evals
        # Deviation 1: always repair before evaluation (code: only in 2nd half).
        arxvalid = self._han_boun(arx.T, x_old, 2).T

        raw = self._eval(arxvalid.T)
        idx = np.argsort(raw, kind="stable")
        raw = raw[idx]
        arxvalid, arx, arz = arxvalid[:, idx], arx[:, idx], arz[:, idx]

        mu = s["mu"]
        if self.cma_weights == "code":
            w = raw[:mu].copy()
            if w.sum() > 1e25 or not np.isfinite(w.sum()) or w.sum() <= 0:
                w = np.ones(mu) / mu
            w = w / w.sum()
        else:
            w = np.ones(mu) / mu
        s["weights"] = w
        s["xold"] = s["xmean"].copy()
        s["xmean"] = arx[:, :mu] @ w
        if second_half:
            s["xmean"] = self._han_boun(s["xmean"][None, :], x_old[0][None, :], 2)[0]
        zmean = arz[:, :mu] @ w
        cs, cc, mueff, chiN = s["cs"], s["cc"], s["mueff"], s["chiN"]
        s["ps"] = (1 - cs) * s["ps"] + math.sqrt(cs * (2 - cs) * mueff) * (s["B"] @ zmean)
        hsig = (np.linalg.norm(s["ps"]) / math.sqrt(1 - (1 - cs) ** (2 * it)) / chiN
                < 1.4 + 2 / (n + 1))
        s["pc"] = (1 - cc) * s["pc"] + hsig * (math.sqrt(cc * (2 - cc) * mueff) / s["sigma"]) * (s["xmean"] - s["xold"])
        c1, cmu = s["ccov1"], s["ccovmu"]
        arpos = (arx[:, :mu] - s["xold"][:, None]) / s["sigma"]
        s["C"] = ((1 - c1 - cmu) * s["C"] + c1 * np.outer(s["pc"], s["pc"])
                  + cmu * (arpos * w[None, :]) @ arpos.T)
        s["sigma"] = s["sigma"] * math.exp(min(1.0, (np.linalg.norm(s["ps"]) / chiN - 1) * cs / s["damps"]))

        healthy = np.all(np.isfinite(s["C"])) and np.isfinite(s["sigma"]) and s["sigma"] > 0
        if healthy and (it % (1 / (c1 + cmu) / n / 10)) < 1:
            C = np.triu(s["C"]) + np.triu(s["C"], 1).T
            try:
                dD, B = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                healthy = False
            else:
                if dD.min() <= 0:
                    dD[dD < 0] = 0
                    tmp = dD.max() / 1e14
                    C = C + tmp * np.eye(n)
                    dD = dD + tmp
                if dD.max() > 1e14 * dD.min():
                    tmp = dD.max() / 1e14 - dD.min()
                    C = C + tmp * np.eye(n)
                    dD = dD + tmp
                s["C"], s["B"] = C, B
                s["diagD"] = np.sqrt(dD)
                s["BD"] = B * s["diagD"][None, :]
        st["EA_2"] = arxvalid.T.copy()
        st["obj2"] = raw.copy()
        if not healthy:  # Deviation 5: numerical guard (not in the code)
            st["cma"] = self._init_cma(st["EA_2"], n, P)

    def _ls2(self, x0: np.ndarray, f0: float):
        """LS2.m — SQP local search from the best-so-far (SLSQP stand-in)."""
        from scipy.optimize import minimize

        budget = int(math.ceil(self.ls_budget_frac * self._max_evals))
        best = [f0, x0.copy()]
        used = [0]

        class _Stop(Exception):
            pass

        def fun(z):
            if used[0] >= budget:
                raise _Stop()
            z = np.clip(np.asarray(z, dtype=float), self._lo, self._hi)
            f = float(self._eval(z[None, :])[0])
            used[0] += 1
            if f < best[0]:
                best[0], best[1] = f, z.copy()
            return f

        try:
            minimize(fun, x0, method="SLSQP",
                     bounds=list(zip(self._lo, self._hi)),
                     options={"maxiter": 10 ** 6, "ftol": 1e-12})
        except _Stop:
            pass
        if f0 - best[0] > 0:
            return best[1], best[0], True
        return x0, f0, False

    # ------------------------------------------------------------- optimize
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        n = self.dim
        lo, hi = self.bounds
        self._lo = np.full(n, float(lo))
        self._hi = np.full(n, float(hi))
        self._hx: list[np.ndarray] = []
        self._hf: list[float] = []
        self._max_evals = max_evals
        history_pop: list[np.ndarray] = []

        PS1 = int(self.pop_size1)
        PS2 = 4 + int(math.floor(3 * math.log(n)))
        PS = PS1 + PS2
        Gmax = (self.g_max if self.g_max is not None
                else self._gmax_default * max_evals / (10000 * n))
        st: dict = {"Gmax": max(Gmax, 1.0)}
        CS = self.cycle
        prob_ls = self.prob_ls
        InitPop = PS1
        try:
            x = self._lo + (self._hi - self._lo) * rng.random((PS, n))
            fitx = self._eval(x)
            b = int(np.argmin(fitx))
            bestold, bestx = float(fitx[b]), x[b].copy()
            st["EA_1"], st["obj1"] = x[:PS1].copy(), fitx[:PS1].copy()
            st["EA_2"], st["obj2"] = x[PS1:].copy(), fitx[PS1:].copy()
            st["cma"] = self._init_cma(st["EA_2"], n, PS2)
            st["probDE1"] = 0.5 * np.ones(2)
            st["probSC"] = 0.5 * np.ones(2)
            st["archNP"] = self.arch_rate * PS1
            st["archive"] = np.zeros((0, n))
            st["hist_pos"] = 0
            M = self.memory_size
            st["af"] = 0.7 * np.ones(M)
            st["acr"] = 0.5 * np.ones(M)
            st["aT"] = 0.1 * np.ones(M)
            st["afreq"] = 0.5 * np.ones(M)
            history_pop.append(x.copy())

            it = 0
            cy = 0
            indx = 0
            Probs = np.ones(2)
            while len(self._hf) < max_evals:
                it += 1
                cy += 1
                if cy == math.ceil(CS + 1):
                    qual = np.array([st["obj1"][0], st["obj2"][0]])
                    sq = qual.sum()
                    norm_qual = 1 - (qual / sq if sq != 0 and np.isfinite(sq) else 0.5 * np.ones(2))
                    D = np.array([
                        np.linalg.norm(st["EA_1"][1:] - st["EA_1"][0], axis=1).mean(),
                        np.linalg.norm(st["EA_2"][1:] - st["EA_2"][0], axis=1).mean()])
                    sd = D.sum()
                    norm_div = D / sd if sd > 0 else 0.5 * np.ones(2)
                    Probs = norm_qual + norm_div
                    Probs = np.fmax(0.1, np.fmin(0.9, Probs / Probs.sum()))
                    indx = int(np.argmax(Probs)) + 1
                    if Probs[0] == Probs[1]:
                        indx = 0
                elif cy == 2 * math.ceil(CS):
                    if indx == 1:
                        k = min(PS2, PS1)
                        li = rng.permutation(PS1)[:k]
                        st["EA_2"][:k] = st["EA_1"][li]
                        st["obj2"][:k] = st["obj1"][li]
                        st["cma"] = self._init_cma(st["EA_2"], n, PS2)
                        st["cma"]["sigma"] *= (1 - len(self._hf) / max_evals)
                    else:
                        e = st["EA_2"][0]
                        if e.min() > lo and e.max() < hi:
                            st["EA_1"][PS1 - 1] = e
                            st["obj1"][PS1 - 1] = st["obj2"][0]
                            o = np.argsort(st["obj1"], kind="stable")
                            st["EA_1"], st["obj1"] = st["EA_1"][o], st["obj1"][o]
                    cy = 1
                    Probs = np.ones(2)

                # ---- perching and patrolling (EBO) ----
                if len(self._hf) < max_evals and rng.random() < Probs[0]:
                    cur = len(self._hf)
                    Upd = _matlab_round(((self.min_pop_size - InitPop) / max_evals) * cur + InitPop)
                    if PS1 > Upd:
                        red = PS1 - Upd
                        if PS1 - red < self.min_pop_size:
                            red = PS1 - self.min_pop_size
                        PS1 -= red
                        st["EA_1"], st["obj1"] = st["EA_1"][:PS1], st["obj1"][:PS1]
                        st["archNP"] = _matlab_round(self.arch_rate * PS1)
                        if len(st["archive"]) > st["archNP"]:
                            st["archive"] = st["archive"][rng.permutation(len(st["archive"]))[:st["archNP"]]]
                    self._ebo(rng, st, it)
                    if st["obj1"][0] < bestold and st["EA_1"][0].min() >= lo and st["EA_1"][0].max() <= hi:
                        bestold, bestx = float(st["obj1"][0]), st["EA_1"][0].copy()

                # ---- scout / CMAR ----
                if len(self._hf) < max_evals and rng.random() < Probs[1]:
                    self._scout(rng, st, it)
                    if st["obj2"][0] < bestold:
                        bestold, bestx = float(st["obj2"][0]), st["EA_2"][0].copy()

                # ---- LS2 (SQP) ----
                if len(self._hf) > 0.75 * max_evals and rng.random() < prob_ls:
                    bestx, bestold, succ = self._ls2(bestx, bestold)
                    if succ:
                        st["EA_1"][PS1 - 1] = bestx
                        st["obj1"][PS1 - 1] = bestold
                        o = np.argsort(st["obj1"], kind="stable")
                        st["EA_1"], st["obj1"] = st["EA_1"][o], st["obj1"][o]
                        st["EA_2"] = np.tile(st["EA_1"][0], (PS2, 1))
                        st["cma"] = self._init_cma(st["EA_2"], n, PS2)
                        st["cma"]["sigma"] = 1e-5
                        st["obj2"] = np.full(PS2, st["obj1"][0])
                        prob_ls = 0.1
                    else:
                        prob_ls = 0.01
                history_pop.append(np.vstack([st["EA_1"], st["EA_2"]]))
        except _BudgetExhausted:
            pass

        final = (np.vstack([st["EA_1"], st["EA_2"]]) if "EA_1" in st else None)
        if final is not None:
            history_pop.append(final)
        return self._make_result(self._hx, self._hf, history_pop)
