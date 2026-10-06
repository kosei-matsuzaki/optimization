"""iCMAES-ILS — IPOP-CMA-ES and an iterated Mtsls1 local search compete, then
the winner takes the rest of the budget (Liao & Stützle, IEEE CEC 2013 winner).

T. Liao and T. Stützle, "Benchmark results for a simple hybrid algorithm on the
CEC 2013 benchmark set for real-parameter optimization", Proc. IEEE CEC 2013,
pp. 1938-1944. The CEC 2013 paper itself was not readable here; the algorithm
and its default parameters are taken from the same authors' description in
T. Liao, "Population-based Heuristic Algorithms for Continuous and Mixed
Discrete-Continuous Optimization Problems", PhD thesis, ULB/IRIDIA 2013,
Ch. 4 (Alg. 3, Alg. 4, Table 4.1), and Sec. 2.1.1 / 2.3.2 for Mtsls1 and
IPOP-CMA-ES.

Algorithm (thesis Alg. 4)
-------------------------
  CompBudget = compr · TotalBudget
  S      := uniform random initial solution
  Sbest  := IPOP-CMA-ES(S, CompBudget)                 # competition, part 1
  S'best := ILS(S, Sbest, CompBudget)                  # competition, part 2
  if f(S'best) < f(Sbest): ILS(S'best, S'best, rest)   # deployment
  else:                    IPOP-CMA-ES(Sbest, rest)    # restarted with default
                                                       # λ0 / σ0, from Sbest
ILS (thesis Alg. 3): Snew = Mtsls1(S); if f(Snew) < f(Sbest): S = Sbest = Snew
(refine, no perturbation) else S = Srand + r·(Sbest − Srand), r ~ U[0,1).
Because Sbest of the competition ILS is IPOP-CMA-ES's best, the ILS acceptance
and perturbation are biased by what IPOP-CMA-ES found (the "cooperative" part).

Mtsls1 (Tseng & Chen 2008, as described in thesis Sec. 2.1.1): per iteration,
for each dimension i in fixed order try x_i − ss, then x_i + 0.5·ss, keep the
first improvement; if a whole iteration improves nothing, halve ss. Stops after
LSIterations iterations.

Parameters (thesis Table 4.1, defaults; B − A = box width)
  IPOP-CMA-ES  λ0 = 4 + floor(3 ln D), µ = floor(λ/2), σ0 = 0.5·(B − A),
               IPOP factor 2, stopTolFun 1e-12, stopTolFunHist 1e-20,
               stopTolX 1e-12; restarts draw a uniform x0 (thesis Alg. 2)
  ILS          LSIterations = 1.5·D, Mtsls1 ss0 = 0.5·(B − A)
  competition  compr = 0.1

Guesses / deviations (not specified in what we could read)
  * Mtsls1 step size across ILS iterations: reset to ss0 after a perturbation,
    carried over (last ss of the previous call) on a refinement call. A reset on
    refinement would restart every refinement at half the box width.
  * LSIterations = 1.5·D rounded up (D = 5 → 8).
  * Tuning table lists a "BiasExtent" parameter defaulting to 0; its meaning is
    not given, so it is omitted (r ~ U[0,1) as in the text).
  * Bound handling: the thesis clamps each solution before evaluation. Mtsls1
    and ILS points are clamped; IPOP-CMA-ES uses pycma's own `bounds`
    (BoundTransform), as the repo's IPOPCMAESOptimizer does. pycma keeps its
    other default stopping criteria (tolstagnation, noeffectaxis, ...).
  * The CMA-ES is pycma's default (incl. its default µ/weights), not the
    authors' C implementation.
"""
from __future__ import annotations

import math

import cma
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _PhaseEnd(Exception):
    """Raised when the current phase's evaluation limit is reached."""


class ICMAESILSOptimizer(BaseOptimizer):
    """iCMAES-ILS (Liao & Stützle, CEC 2013 winner)."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        compr: float = 0.1,            # Table 4.1
        sigma0_frac: float = 0.5,      # σ0 = 0.5 (B − A), Table 4.1
        incpopsize: int = 2,           # IPOP factor, Table 4.1
        tolfun: float = 1e-12,         # Table 4.1
        tolfunhist: float = 1e-20,     # Table 4.1
        tolx: float = 1e-12,           # Table 4.1
        ls_iter_factor: float = 1.5,   # LSIterations = 1.5 D, Table 4.1
        ss_frac: float = 0.5,          # Mtsls1 ss0 = 0.5 (B − A), Sec. 2.1.1
    ):
        super().__init__(benchmark, seed)
        self.compr = compr
        self.sigma0_frac = sigma0_frac
        self.incpopsize = incpopsize
        self.tolfun = tolfun
        self.tolfunhist = tolfunhist
        self.tolx = tolx
        self.ls_iter_factor = ls_iter_factor
        self.ss_frac = ss_frac

    # ------------------------------------------------------------------ helpers
    def _eval(self, x: np.ndarray) -> float:
        if len(self._hf) >= self._limit:
            raise _PhaseEnd()
        x = np.clip(np.asarray(x, dtype=float), self._lo, self._hi)
        f = float(self.func(x))
        self._hx.append(x.copy())
        self._hf.append(f)
        return f

    def _ipop(self, x0: np.ndarray, rng) -> tuple[np.ndarray, float]:
        """IPOP-CMA-ES from x0 with default λ0/σ0 until the phase limit."""
        D = self.dim
        lam0 = 4 + int(math.floor(3 * math.log(D)))
        sigma0 = self.sigma0_frac * (self._hi - self._lo)
        best_x, best_f = x0.copy(), math.inf
        restart = 0
        try:
            while True:
                remaining = self._limit - len(self._hf)
                if remaining <= 0:
                    break
                xs = x0 if restart == 0 else rng.uniform(self._lo, self._hi, D)
                opts = cma.CMAOptions()
                opts["seed"] = int(rng.integers(1, 2**31 - 1))
                opts["bounds"] = [[self._lo] * D, [self._hi] * D]
                opts["popsize"] = lam0 * self.incpopsize ** restart
                opts["tolfun"] = self.tolfun
                opts["tolfunhist"] = self.tolfunhist
                opts["tolx"] = self.tolx
                opts["maxfevals"] = remaining
                opts["verbose"] = -9
                es = cma.CMAEvolutionStrategy(np.asarray(xs, dtype=float), sigma0, opts)
                while not es.stop():
                    sols = es.ask()
                    fits = []
                    for s in sols:
                        f = self._eval(s)
                        fits.append(f)
                        if f < best_f:
                            best_f, best_x = f, np.clip(np.asarray(s), self._lo, self._hi)
                    es.tell(sols, fits)
                    self._hp.append(np.clip(np.array(sols), self._lo, self._hi))
                restart += 1
        except _PhaseEnd:
            pass
        return best_x, best_f

    def _ils(self, S, fS, Sbest, fbest, rng):
        """ILS until the phase limit; returns its best (starts at Sbest)."""
        ss0 = self.ss_frac * (self._hi - self._lo)
        iters = max(1, int(math.ceil(self.ls_iter_factor * self.dim)))
        ss = ss0
        S = S.copy()
        try:
            if fS is None:
                fS = self._eval(S)
            while True:
                # _mtsls1 keeps its running best in self._ls_best so a phase
                # end mid-call still credits the improvement found so far.
                Snew, fnew, ss_end = self._mtsls1(S, fS, ss, iters)
                if fnew < fbest:
                    S, fS = Snew, fnew
                    Sbest, fbest = Snew.copy(), fnew
                    ss = ss_end
                else:
                    r = float(rng.random())
                    Srand = rng.uniform(self._lo, self._hi, self.dim)
                    S = Srand + r * (Sbest - Srand)
                    fS = self._eval(S)
                    ss = ss0
        except _PhaseEnd:
            if self._ls_best is not None and self._ls_best[1] < fbest:
                Sbest, fbest = self._ls_best
        return Sbest, fbest

    def _mtsls1(self, x, fx, ss, iters):
        self._ls_best = (x.copy(), fx)
        x = x.copy()
        for _ in range(iters):
            improved = False
            for i in range(self.dim):
                xi = x[i]
                for cand in (xi - ss, xi + 0.5 * ss):
                    x[i] = min(max(cand, self._lo), self._hi)
                    f = self._eval(x)
                    if f < fx:
                        fx, improved = f, True
                        self._ls_best = (x.copy(), fx)
                        break
                else:
                    x[i] = xi
            if not improved:
                ss *= 0.5
        self._ls_best = None
        return x, fx, ss

    # ----------------------------------------------------------------- optimize
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        self._lo, self._hi = float(lo), float(hi)
        self._hx, self._hf, self._hp = [], [], []
        self._ls_best = None

        comp = max(1, int(self.compr * max_evals))
        S = rng.uniform(self._lo, self._hi, self.dim)

        # Competition part 1: IPOP-CMA-ES from S for CompBudget evaluations.
        self._limit = min(comp, max_evals)
        Sbest, fbest = self._ipop(S, rng)

        # Competition part 2: ILS from S, acceptance biased by IPOP's best.
        self._limit = min(2 * comp, max_evals)
        Sb2, fb2 = self._ils(S, None, Sbest, fbest, rng)

        # Deployment: the better of the two gets the rest of the budget.
        self._limit = max_evals
        self.winner_ = "ILS" if fb2 < fbest else "IPOP-CMA-ES"
        if fb2 < fbest:
            self._ils(Sb2, fb2, Sb2, fb2, rng)
        else:
            self._ipop(Sbest, rng)

        return self._make_result(self._hx, self._hf, self._hp or None)
