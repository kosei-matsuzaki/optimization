"""CoBiDE — standalone numpy port (Wang, Li, Huang & Li, 2014).

    Y. Wang, H.-X. Li, T. Huang and L. Li, "Differential evolution based on
    covariance matrix learning and bimodal distribution parameter setting",
    Applied Soft Computing 18 (2014) 232-247.

Sources (short tags):

* [P]    the paper (author copy intleo.csu.edu.cn/codes/CoBiDE.pdf): Sec. 4,
         Eqs. (8)-(14), Fig. 2 (pseudo-code), Sec. 5 (NP = 60, pb = 0.4,
         ps = 0.5), Sec. 5.4 (pb recommended in [0.2, 0.7], ps in [0.3, 0.6]).
* [CODE] the authors' MATLAB ``CoBiDE.m`` (Y. Wang & L. Li, 21/11/2013),
         intleo.csu.edu.cn/codes/CoBiDE.rar.

Algorithm (per generation; synchronous / generational DE):
  * F_i, CR_i are kept per individual. An individual whose trial succeeded
    (f(u) <= f(x)) keeps its pair; otherwise a new pair is drawn ([P] Fig. 2
    step 26-28; [CODE] chgF = ~S).
      F  ~ Cauchy(0.65, 0.1) or Cauchy(1.0, 0.1) w.p. 1/2; > 1 -> 1, <= 0 ->
           redrawn (from the same mode in [CODE]).                 [P] Eq. 13
      CR ~ Cauchy(0.1, 0.1) or Cauchy(0.95, 0.1) w.p. 1/2; > 1 -> 1, < 0 -> 0
           ([P] Eq. 14).  [CODE] sets CR <= 0 to 0.1 instead -> option
           ``cr_low_value`` (default 0.0 = paper).
  * DE/rand/1 mutant; r1, r2, r3 distinct and != i.
  * mutant repair ([CODE]): a coordinate below lo is reflected (2 lo - v); if
    that overshoots hi it is set to hi (symmetric for the upper bound).
  * with probability pb (once per generation, for the whole population)
    covariance matrix learning ([P] Eqs. 8-12): C = sample covariance of the
    best round(ps NP) individuals; C = B D^2 B^T; if cond(C) > 1e20, C += (max
    eig / 1e20 - min eig) I ([CODE]); binomial crossover (one guaranteed
    coordinate) on B^T x and B^T v; trial = B u'; the trial is then repaired
    as above. Otherwise ordinary binomial crossover (no repair needed: both
    parents are inside).
  * selection u replaces x if f(u) <= f(x) (whole generation, then replace).

Parameters:
  pop_size NP 60   [P] Sec. 5 (CEC 2005, D = 30); the paper uses 60 for all
                   functions; it is not scaled with D. (Guess: kept at 60 for
                   other D.)
  pb 0.4, ps 0.5   [P] Sec. 5 / [CODE]
  cr_low_value 0.0 [P]; [CODE] uses 0.1.

Differences from the CoBiDE component of EA4eig (``ea4eig.py``, Bujok &
Kolenovsky's transcription):
  * EA4eig clips CR < 0 to 0 (= [P]); [CODE] uses 0.1 (option here).
  * EA4eig redraws the Cauchy *mode* on every F <= 0 retry; [CODE] retries
    within the mode first chosen (done here).
  * EA4eig forces a mutant coordinate only when the binomial mask is empty;
    [CODE] always forces coordinate jrand (done here) — slightly higher
    effective CR.
  * EA4eig's bound handling is full mirroring (zrcad) of mutant and trial;
    CoBiDE reflects once and clamps to the opposite bound.
  * EA4eig's CoBiDE has no cond(C) <= 1e20 safeguard.
  * EA4eig uses ``eig(cov(...))`` of the shared EA4eig population and the
    eigen crossover's CR also feeds IDEbd / jSO; standalone there is no
    population sharing, no LPSR (fixed NP = 60) and no roulette selection.

Deviations: the budget is checked per evaluation (the last generation is cut
at ``max_evals``; selection is then applied to the evaluated prefix only).
RNG streams differ from MATLAB. [CODE]'s special case for CEC 2005 F7/F25
(no bounds) does not apply.
"""
from __future__ import annotations

import math

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


def _mround(x: float) -> int:
    return int(math.floor(x + 0.5))


class CoBiDEOptimizer(BaseOptimizer):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int = 60,
        pb: float = 0.4,
        ps: float = 0.5,
        cr_low_value: float = 0.0,
    ):
        super().__init__(benchmark, seed)
        self.pop_size = int(pop_size)
        self.pb = pb
        self.ps = ps
        self.cr_low_value = cr_low_value

    # bimodal samplers ([P] Eqs. 13-14)
    def _draw_F(self, rng: np.random.Generator) -> float:
        loc = 0.65 if rng.random() < 0.5 else 1.0
        while True:
            f = loc + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
            f = min(1.0, f)
            if f > 0.0:
                return f

    def _draw_CR(self, rng: np.random.Generator) -> float:
        loc = 0.1 if rng.random() < 0.5 else 0.95
        c = loc + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
        if c <= 0.0:
            return self.cr_low_value
        return min(c, 1.0)

    @staticmethod
    def _repair(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """[CODE]: reflect once at the violated bound, clamp to the other bound
        if the reflection overshoots."""
        y = y.copy()
        bl = y < lo
        y[bl] = 2 * lo[bl] - y[bl]
        y[bl] = np.minimum(y[bl], hi[bl])
        bu = y > hi
        y[bu] = 2 * hi[bu] - y[bu]
        y[bu] = np.maximum(y[bu], lo[bu])
        return y

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        d = self.dim
        lo = np.broadcast_to(np.asarray(self.bounds[0], dtype=float), (d,)).copy()
        hi = np.broadcast_to(np.asarray(self.bounds[1], dtype=float), (d,)).copy()
        hx: list[np.ndarray] = []
        hf: list[float] = []
        hpop: list[np.ndarray] = []

        def evaluate(x: np.ndarray) -> float:
            f = float(self.func(x))
            hx.append(x.copy())
            hf.append(f)
            return f

        NP = min(self.pop_size, max_evals)
        X = lo + rng.random((NP, d)) * (hi - lo)
        fit = np.array([evaluate(x) for x in X])
        hpop.append(X.copy())
        if NP < 4:
            return self._make_result(hx, hf, hpop)
        sel_n = max(2, _mround(self.ps * NP))

        F = np.array([self._draw_F(rng) for _ in range(NP)])
        CR = np.array([self._draw_CR(rng) for _ in range(NP)])
        S = np.ones(NP, dtype=bool)

        while len(hf) < max_evals:
            for i in np.flatnonzero(~S):
                F[i] = self._draw_F(rng)
                CR[i] = self._draw_CR(rng)

            V = np.empty((NP, d))
            for i in range(NP):
                cand = rng.permutation(NP - 1)[:3]
                r1, r2, r3 = (c + (c >= i) for c in cand)
                V[i] = self._repair(X[r1] + F[i] * (X[r2] - X[r3]), lo, hi)

            jrand = rng.integers(0, d, NP)
            crs = rng.random((NP, d)) < CR[:, None]
            crs[np.arange(NP), jrand] = True

            if rng.random() < self.pb:
                top = X[np.argsort(fit, kind="stable")[:sel_n]]
                C = np.atleast_2d(np.cov(top, rowvar=False))
                C = (C + C.T) / 2.0
                if np.all(np.isfinite(C)):
                    ev, R = np.linalg.eigh(C)
                    if ev.max() > 1e20 * ev.min():
                        C = C + (ev.max() / 1e20 - ev.min()) * np.eye(d)
                        ev, R = np.linalg.eigh(C)
                else:
                    R = np.eye(d)
                Xr = X @ R
                Vr = V @ R
                U = np.where(crs, Vr, Xr) @ R.T
                U = np.array([self._repair(u, lo, hi) for u in U])
            else:
                U = np.where(crs, V, X)

            n_eval = min(NP, max_evals - len(hf))
            fu = np.array([evaluate(U[i]) for i in range(n_eval)])
            S = np.zeros(NP, dtype=bool)
            S[:n_eval] = fu <= fit[:n_eval]
            X[S] = U[S]
            fit[S] = fu[S[:n_eval]]
            hpop.append(X.copy())

        return self._make_result(hx, hf, hpop)
