"""EA4eig — cooperative model of four EAs with Eigen crossover (CEC 2022 winner).

Sources (cited in the docstrings below by these short tags):

* [BK22]  P. Bujok, P. Kolenovsky, "Eigen Crossover in Cooperative Model of
          Evolutionary Algorithms Applied to CEC 2022 Single Objective
          Numerical Optimisation", IEEE CEC 2022.
* [CODE]  The authors' official MATLAB implementation ``Run_EA4eig.m`` from the
          competition repository github.com/P-N-Suganthan/2022-SO-BO, as
          redistributed (with the bound-handling fix, see below) in folder
          ``CEC2022_EA4eig_CORRECTED`` of Biedrzycki's supplementary material
          (staff.elka.pw.edu.pl/~rbiedrzy/publ/EA4EigSimpl.zip). Every constant
          below was transcribed from this file; line numbers refer to it.
* [RB]    R. Biedrzycki, "Analysis and simplification of the winner of the CEC
          2022 optimization competition on single objective bound constrained
          search", Evolutionary Computation (author's version,
          staff.elka.pw.edu.pl/~rbiedrzy/publ/Ea4EigSimplifying.pdf):
          Fig. 2 (EA4Eig), Fig. 3 (IDEbd), Sec. 3.1 / 3.3 (bugs), Table 5
          (parameter defaults / tuned values).
* [RBCPP] Biedrzycki's C++ code in the same zip: ``EA4EIGjSO_IDE/`` (EA4Eig
          restricted to jSO + IDEbd) and
          ``onlyIDEcorrItMaxWithoutSortsNoIDEbdmutNoRandxr3CleaningWithoutFails/``
          (the final simplified algorithm).

The [BK22] paper itself was not available to the implementer; [CODE] is the
reference, cross-checked against the description in [RB] Sec. 2.3-2.4.

Two classes are provided (plus a preset):

* ``EA4eigOptimizer`` — the full algorithm (CoBiDE, IDEbd, CMA-ES, jSO on one
  shared population). ``components`` selects a subset, e.g.
  ``("idebd", "jso")`` is the best variant of [RB] Table 2 / the 40-D winner
  of [RB] Table 11 (``EA4eigJsoIdebdOptimizer`` presets it).
* ``EA4eigSimplifiedOptimizer`` — the final simplification of [RB] (IDEbd only,
  corrected t_max and r1 selection, single mutation, no x_r3 perturbation,
  no stage-switch failure counter) — [RBCPP] ``onlyIDE...WithoutFails``.

Budget: every objective call is appended to the history; the run stops
exactly at ``max_evals`` (the MATLAB code evaluates whole generations and may
overshoot; here a generation is cut short). Unlike [CODE] the run does not
stop when the error reaches 1e-8 (the optimum is not known to a black-box
optimiser in this repository's protocol).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    pass


class _RunStopped(Exception):
    """[CODE] l.534 `break` — the CMA-ES branch terminates the whole run."""


def _mround(x: float) -> int:
    """MATLAB ``round`` (half away from zero) for non-negative arguments."""
    return int(math.floor(x + 0.5))


def _mirror(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """[CODE] ``zrcad.m``: reflect out-of-range coordinates at the violated bound
    until inside. Repeated reflection between two walls equals folding with
    period 2(hi-lo), which is what is computed here (no infinite loop)."""
    out = (y < lo) | (y > hi)
    if not out.any():
        return y
    y = y.copy()
    w = hi - lo
    t = np.mod(y[out] - lo, 2.0 * w)
    y[out] = lo + np.where(t > w, 2.0 * w - t, t)
    return y


def _rand_excl(rng: np.random.Generator, n: int, k: int,
               excl: Sequence[int]) -> list[int]:
    """[CODE] ``nahvyb_expt.m``: k distinct indices from range(n), none in excl."""
    taken = set(int(e) for e in excl)
    out: list[int] = []
    while len(out) < k:
        c = int(rng.integers(n))
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out


def _eig_vectors(pop_sorted: np.ndarray, n_keep: int) -> np.ndarray:
    """Eigenvectors of the sample covariance of the best ``n_keep`` rows
    ([CODE] l.193-195, ``eig(cov(...))``; [RB] Sec. 2.3)."""
    sub = pop_sorted[:max(n_keep, 1)]
    if sub.shape[0] < 2:
        return np.eye(pop_sorted.shape[1])
    c = np.cov(sub, rowvar=False)
    c = np.atleast_2d(c)
    if not np.all(np.isfinite(c)):
        return np.eye(pop_sorted.shape[1])
    _, vec = np.linalg.eigh(c)
    return vec


def _eig_cross(rng: np.random.Generator, E: np.ndarray, x: np.ndarray,
               v: np.ndarray, cr: float) -> np.ndarray:
    """Eigen crossover (Guo & Yang 2015) as in [CODE] l.207-214: binomial
    crossover in the eigen-coordinate system, >=1 coordinate from the mutant."""
    d = x.size
    xe = E.T @ x
    ve = E.T @ v
    change = rng.random(d) < cr
    if not change.any():
        change[int(rng.integers(d))] = True
    xe[change] = ve[change]
    return E @ xe


def _cauchy(rng: np.random.Generator, x0: float, gamma: float) -> float:
    """[CODE] ``cauchy_rnd.m``."""
    return x0 + gamma * math.tan(math.pi * (rng.random() - 0.5))


class EA4eigOptimizer(BaseOptimizer):
    """EA4eig (Bujok & Kolenovsky, CEC 2022 winner) — full algorithm.

    One population P (size N) is shared by up to four component algorithms:
    1 CoBiDE, 2 IDEbd, 3 CMA-ES, 4 jSO ([CODE] l.170). Every generation one
    component h is drawn by roulette with probability n_h / sum(n) where n_h
    is its success count (initial n0); if the smallest probability falls below
    delta = 1/(5 H) (H = number of components) all counts are reset to n0
    ([CODE] l.175-186, ``roulete.m``; [RB] Fig. 2 with p_min = 1/20).
    Success = trial accepted by DE selection (<=; jSO counts only strict <)
    or, for CMA-ES, a sample better than the current worst member, which it
    then replaces. DE components use Eigen crossover with probability p_eig
    per generation (eigenvectors of the covariance of the best
    round(N*eig_pop_ratio) members), binomial otherwise. After every
    generation N is reduced linearly from ``pop_size`` to ``min_pop_size``
    by evaluations (L-SHADE LPSR, [CODE] l.766-794; worst members removed).

    Bound handling (as in [CODE]):
      * CoBiDE: mutant and trial are reflected at the bounds (``zrcad``).
      * IDEbd, binomial branch: an out-of-range coordinate is re-drawn
        uniformly in [lo, hi] (l.413-415); Eigen branch: reflection.
      * jSO: reflection (both branches; see ``fix_eig_bounds``).
      * CMA-ES: an out-of-range coordinate is set to the midpoint between the
        bound and the same coordinate of the k-th sample of the previous
        CMA-ES generation (l.471-475; initially the k-th initial member).

    Parameters (default = value in [CODE]; [RB] Table 5 lists the same
    defaults for the shared ones):
      pop_size=100        N_init, [CODE] l.51; [RB] Table 5 "mu".
      min_pop_size=10     N_min, [CODE] l.53; [RB] Table 5 "mu_min".
      n0=2                initial/reset success count, [CODE] l.54; [RB] Fig.2.
      delta=None          roulette reset threshold; None -> 1/(5 H),
                          [CODE] l.74 (=1/20 for H=4, [RB] Fig.2 p_min).
      p_eig=0.4           Eigen-crossover probability per generation,
                          [CODE] l.87; [RB] Table 5.
      eig_pop_ratio=0.5   share of best members used for the covariance,
                          [CODE] l.86 "CBps"; [RB] Table 5 "mu ratio for Eig.".
      components=("cobide","idebd","cmaes","jso")  subset to run ([RB] Tab. 2).
      Bug switches (documented by [RB]; default = corrected behaviour):
      fix_eig_bounds=True   [RB] Sec. 3.1: the jSO + Eigen-crossover path of
                          the competition code skipped the bound repair, so
                          infeasible points were evaluated. False reproduces
                          the competition submission.
      fix_idebd_r1=True   [RB] Sec. 3.3 "corr. r1 selection": the code meant
                          to draw r1 from the superior members excluding the
                          indices already used, but drew a raw position in
                          1..len(candidates). False reproduces [CODE]
                          (and Biedrzycki's "CORRECTED" EA4Eig, which keeps it).
      fix_idebd_tmax=True [RB] Sec. 3.3 "corr. t_max": when the IDEbd
                          generation counter exceeds t_max (fixed at
                          maxFES/N_init), t_max is set to it. False = [CODE].
      cma_cond_stop=False [CODE] l.534 inherits purecmaes' `break`, which in
                          EA4eig terminates the *whole run* when the CMA-ES
                          condition number exceeds 1e14 (max D > 1e7 min D).
                          Not discussed in [RB]; treated here as unintended
                          and disabled by default. True reproduces it (the
                          companion test ``PopFit(1) <= 1e-6`` on the raw,
                          bias-carrying CEC value is omitted because it
                          depends on the CEC bias convention).

    Component constants (fixed, from [CODE]):
      CoBiDE: F ~ Cauchy(0.65 or 1.0 w.p. 1/2, 0.1), redrawn while F<0,
        capped at 1; CR ~ Cauchy(0.1 or 0.95 w.p. 1/2, 0.1) clipped to [0,1];
        one (F, CR) pair per population slot, redrawn when that slot's trial
        fails (l.93-121, 255-281). DE/rand/1. The per-slot CR is also the CR
        of Eigen crossover in IDEbd and jSO (l.392, 611; [RB] Sec. 2.3).
      IDEbd ([RB] Fig. 3): t_max = round(maxFES/N_init); stage-2 threshold
        t_th = fix(t_max/2); fails_th = t_max/10; ps = 0.1 + 0.9 *
        10^(5 (g/t_max - 1)); pd = 0.1 ps; F ~ N(o/N, 0.1), CR ~ N(i/N, 0.1)
        (1-based ranks, redrawn until in range); SRT = 0 / 0.1 (l.76-83,
        339-379, 404-406).
      CMA-ES: sigma0 = (hi-lo)/2; mu = floor(N/2) log weights; standard
        Hansen (2001) constants computed once from the initial N; eigen
        update every N/(c1+cmu)/D/10 evaluations; sigma reset to (hi-lo)/2 if
        it leaves [1e-300, 1e300]; the mean is restarted every CMA-ES
        generation from the weighted best half of P (l.123-149, 461-536).
      jSO: H=5 memories (M_F=0.3, M_CR=0.8, last slot fixed 0.9); archive
        2.6 N; p linearly 0.25 -> 0.125; CR ~ N(M_CR, sqrt(0.1)) clipped,
        >=0.7 (<25% FES) / >=0.6 (<50% FES); F ~ Cauchy(M_F, 0.1) redrawn
        while <=0, capped at 1 and at 0.7 for FES < 60%; F_w = 0.7F/0.8F/1.2F
        (thresholds 20%/40% FES); weighted-Lehmer memory update averaged
        with the old value; memory index cycles over 1..H-1 (l.151-162,
        538-752 — reproduced literally including the MCR=-1 averaging).

    Guesses / deviations (not specified by any source):
      * CMA-ES: the sigma exponent is capped at 700 (MATLAB overflows to Inf
        and the reset rule fires — same result); if the evolution paths or C
        become non-finite they are reset to 0 / identity (MATLAB would keep
        propagating NaN).
      * CMA-ES covariance eigenvalues are clipped to >= 1e-20 * max before the
        square root (MATLAB would silently go complex on a negative one).
      * If no superior candidate remains for r1 with ``fix_idebd_r1`` the best
        member (index 0) is used, as [RBCPP] does.
      * Random-number streams obviously differ from MATLAB's.
    """

    _ALL = ("cobide", "idebd", "cmaes", "jso")

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int = 100,
        min_pop_size: int = 10,
        n0: int = 2,
        delta: float | None = None,
        p_eig: float = 0.4,
        eig_pop_ratio: float = 0.5,
        components: Sequence[str] = ("cobide", "idebd", "cmaes", "jso"),
        fix_eig_bounds: bool = True,
        fix_idebd_r1: bool = True,
        fix_idebd_tmax: bool = True,
        cma_cond_stop: bool = False,
    ):
        super().__init__(benchmark, seed)
        comps = tuple(c.lower() for c in components)
        bad = [c for c in comps if c not in self._ALL]
        if bad or not comps:
            raise ValueError(f"unknown components {bad}; choose from {self._ALL}")
        # keep the original order 1 CoBiDE, 2 IDEbd, 3 CMA-ES, 4 jSO
        self.components = tuple(c for c in self._ALL if c in comps)
        self.pop_size = int(pop_size)
        self.min_pop_size = int(min(min_pop_size, pop_size))
        self.n0 = n0
        self.delta = (1.0 / (5 * len(self.components))) if delta is None else delta
        self.p_eig = p_eig
        self.eig_pop_ratio = eig_pop_ratio
        self.fix_eig_bounds = fix_eig_bounds
        self.fix_idebd_r1 = fix_idebd_r1
        self.fix_idebd_tmax = fix_idebd_tmax
        self.cma_cond_stop = cma_cond_stop

    # ------------------------------------------------------------------ #
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        D = self.dim
        hx: list[np.ndarray] = []
        hf: list[float] = []
        hpop: list[np.ndarray] = []
        func = self.func
        maxFES = int(max_evals)

        def ev(x: np.ndarray) -> float:
            if len(hf) >= maxFES:
                raise _BudgetExhausted
            x = np.array(x, dtype=float)
            f = float(func(x))
            hx.append(x)
            hf.append(f)
            return f

        try:
            self._run(rng, ev, hf, hpop, lo, hi, D, maxFES)
        except (_BudgetExhausted, _RunStopped):
            pass
        return self._make_result(hx, hf, hpop)

    # ------------------------------------------------------------------ #
    def _run(self, rng, ev, hf, hpop, lo, hi, D, maxFES):
        N_init = self.pop_size
        N = N_init
        Nmin = self.min_pop_size
        comps = self.components
        H_alg = len(comps)
        ni = np.full(H_alg, float(self.n0))
        width = hi - lo

        def fes() -> int:
            return len(hf)

        # --- initial population ([CODE] l.66-68) -----------------------
        P = lo + rng.random((N, D)) * width
        fit = np.empty(N)
        for i in range(N):
            fit[i] = ev(P[i])
        hpop.append(P.copy())

        # --- IDEbd state ([CODE] l.76-83) ------------------------------
        gmax = max(1, _mround(maxFES / N))
        T = gmax / 10.0
        gt = int(gmax / 2)
        g = 0
        Tcurr = 0
        CBps = self.eig_pop_ratio
        p_eig = self.p_eig

        # --- CoBiDE per-slot F / CR ([CODE] l.93-121) ------------------
        def draw_F() -> float:
            while True:
                f = _cauchy(rng, 0.65, 0.1) if rng.random() < 0.5 else _cauchy(rng, 1.0, 0.1)
                if f >= 0:
                    return min(f, 1.0)

        def draw_CR() -> float:
            c = _cauchy(rng, 0.1, 0.1) if rng.random() < 0.5 else _cauchy(rng, 0.95, 0.1)
            return min(max(c, 0.0), 1.0)

        CBF = np.empty(N)
        CBCR = np.empty(N)
        for i in range(N):
            CBF[i] = draw_F()
            CBCR[i] = draw_CR()

        # --- CMA-ES state ([CODE] l.123-149) ---------------------------
        sigma = width / 2.0
        oldPop = P.copy()
        mu_f = N / 2.0
        mu = int(math.floor(mu_f))
        weights = math.log(mu_f + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = weights.sum() ** 2 / np.sum(weights ** 2)
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        pc = np.zeros(D)
        ps = np.zeros(D)
        B = np.eye(D)
        Dd = np.ones(D)
        CC = np.eye(D)
        invsqrtC = np.eye(D)
        eigeneval = 0
        chiN = D ** 0.5 * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

        # --- jSO state ([CODE] l.151-162) ------------------------------
        Asize_max = _mround(N * 2.6)
        Hm = 5
        MF = np.full(Hm, 0.3)
        MCR = np.full(Hm, 0.8)
        MF[Hm - 1] = 0.9
        MCR[Hm - 1] = 0.9
        k = 0
        A = np.empty((0, D))
        pmax = 0.25
        pmin = pmax / 2

        def n_eig_rows(n: int) -> int:
            # Popeig(round(N*CBps+1):N,:) = []  -> keeps round(N*CBps+1)-1 rows
            return _mround(n * CBps + 1) - 1

        def sort_P():
            nonlocal P, fit, CBF, CBCR
            o = np.argsort(fit, kind="stable")
            P, fit, CBF, CBCR = P[o], fit[o], CBF[o], CBCR[o]

        # ================= main loop ([CODE] l.173) =====================
        while fes() < maxFES:
            # roulette ([CODE] roulete.m) then reset rule (l.182-186)
            ssum = ni.sum()
            p_min = ni.min() / ssum
            cp = np.cumsum(ni) / ssum
            hh = int(np.sum(cp < rng.random()))
            hh = min(hh, H_alg - 1)
            if p_min < self.delta:
                ni[:] = self.n0
            alg = comps[hh]

            if alg == "cobide":  # ---------------- [CODE] l.189-332
                Q = np.empty((N, D))
                Qf = np.full(N, np.inf)
                use_eig = rng.random() < p_eig
                if use_eig:
                    o = np.argsort(fit, kind="stable")
                    E = _eig_vectors(P[o], n_eig_rows(N))
                for i in range(N):
                    r1, r2, r3 = _rand_excl(rng, N, 3, [i])
                    v = P[r1] + CBF[i] * (P[r2] - P[r3])
                    v = _mirror(v, lo, hi)
                    if use_eig:
                        y = _eig_cross(rng, E, P[i], v, CBCR[i])
                    else:
                        y = P[i].copy()
                        change = rng.random(D) < CBCR[i]
                        if not change.any():
                            change[int(rng.integers(D))] = True
                        y[change] = v[change]
                    y = _mirror(y, lo, hi)
                    Q[i] = y
                    Qf[i] = ev(y)
                for i in range(N):
                    if Qf[i] <= fit[i]:
                        P[i] = Q[i]
                        fit[i] = Qf[i]
                        ni[hh] += 1
                    else:
                        CBF[i] = draw_F()
                        CBCR[i] = draw_CR()

            elif alg == "idebd":  # --------------- [CODE] l.334-459
                sort_P()
                if self.fix_idebd_tmax and g > gmax:
                    gmax = g
                Q = P.copy()
                IDEps = 0.1 + 0.9 * 10.0 ** (5 * (g / gmax - 1))
                pd = 0.1 * IDEps
                SRT = 0.0 if g < gt else 0.1
                high = int(IDEps * N)
                for i in range(N):
                    vyb = _rand_excl(rng, N, 4, [i])
                    o_ = vyb[0] if g > gt else i
                    r1, r2, r3 = vyb[1], vyb[2], vyb[3]
                    xo = P[o_]
                    xr2 = P[r2]
                    xr3 = P[r3].copy()
                    pert = rng.random(D) < pd
                    pom = lo + rng.random(D) * width
                    xr3[pert] = pom[pert]
                    Fo = (o_ + 1) / N + 0.1 * rng.standard_normal()
                    while Fo <= 0 or Fo > 1:
                        Fo = (o_ + 1) / N + 0.1 * rng.standard_normal()
                    if (o_ + 1) > high and (r1 + 1) > high:
                        used = set(vyb) | {i}
                        cand = [c for c in range(high) if c not in used]
                        if self.fix_idebd_r1:
                            r1 = cand[int(rng.integers(len(cand)))] if cand else 0
                        else:  # [CODE] l.371: r1 = 1 + fix(rand*length(candidates))
                            r1 = int(rng.random() * len(cand))
                    xr1 = P[r1]
                    if g > gt and rng.random() < 0.5:
                        Q[i] = P[i] + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
                    else:
                        Q[i] = xo + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
                if rng.random() < p_eig:
                    E = _eig_vectors(P, n_eig_rows(N))  # P already sorted
                    for i in range(N):
                        Q[i] = _mirror(_eig_cross(rng, E, P[i], Q[i], CBCR[i]), lo, hi)
                else:
                    for i in range(N):
                        CR = (i + 1) / N + 0.1 * rng.standard_normal()
                        while CR < 0 or CR > 1:
                            CR = (i + 1) / N + 0.1 * rng.standard_normal()
                        jrand = int(rng.integers(D))
                        keep = ~(rng.random(D) <= CR)
                        keep[jrand] = False
                        Q[i, keep] = P[i, keep]
                        bad = (Q[i] < lo) | (Q[i] > hi)
                        if bad.any():
                            Q[i, bad] = lo + rng.random(int(bad.sum())) * width
                Qf = np.empty(N)
                for i in range(N):
                    Qf[i] = ev(Q[i])
                succ = Qf <= fit
                ns = int(succ.sum())
                ni[hh] += ns
                SR = ns / N
                if g < gt:
                    Tcurr = Tcurr + 1 if SR <= SRT else 0
                    if Tcurr >= T:
                        gt = g
                P[succ] = Q[succ]
                fit[succ] = Qf[succ]
                sort_P()
                g += 1

            elif alg == "cmaes":  # --------------- [CODE] l.461-536
                sort_P()
                xmean = P[:mu].T @ weights
                Pop = np.empty((N, D))
                PopFit = np.empty(N)
                for kk in range(N):
                    x = xmean + sigma * (B @ (Dd * rng.standard_normal(D)))
                    m = x < lo
                    x[m] = (oldPop[kk, m] + lo) / 2
                    m = x > hi
                    x[m] = (oldPop[kk, m] + hi) / 2
                    Pop[kk] = x
                    PopFit[kk] = ev(x)
                    w = int(np.argmax(fit))
                    if PopFit[kk] < fit[w]:
                        P[w] = x
                        fit[w] = PopFit[kk]
                        ni[hh] += 1
                order = np.argsort(PopFit, kind="stable")
                xold = xmean
                sel = Pop[order[:mu]]
                xmean = sel.T @ weights
                oldPop = Pop
                FES = fes()
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold)) / sigma
                hsig = float(np.sum(ps ** 2) / (1 - (1 - cs) ** (2 * FES / N)) / D
                             < 2 + 4 / (D + 1))
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
                artmp = (sel - xold).T / sigma
                CC = ((1 - c1 - cmu) * CC
                      + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * CC)
                      + cmu * (artmp * weights) @ artmp.T)
                # MATLAB exp overflows to Inf, which the reset below catches
                sigma = sigma * math.exp(min((cs / damps) * (np.linalg.norm(ps) / chiN - 1), 700.0))
                if sigma > 1e300 or sigma < 1e-300 or not np.isfinite(sigma):
                    sigma = width / 2.0
                # guess: MATLAB would carry NaN/Inf forever; restart the paths/C
                if not (np.all(np.isfinite(ps)) and np.all(np.isfinite(pc))
                        and np.all(np.isfinite(CC))):
                    ps, pc = np.zeros(D), np.zeros(D)
                    CC, B, Dd, invsqrtC = np.eye(D), np.eye(D), np.ones(D), np.eye(D)
                if FES - eigeneval > N / (c1 + cmu) / D / 10:
                    eigeneval = FES
                    CC = np.triu(CC) + np.triu(CC, 1).T
                    if np.all(np.isfinite(CC)):
                        ev_, B = np.linalg.eigh(CC)
                        ev_ = np.maximum(ev_, 1e-20 * max(ev_.max(), 1e-300))
                        Dd = np.sqrt(ev_)
                        invsqrtC = B @ np.diag(1.0 / Dd) @ B.T
                if self.cma_cond_stop and Dd.max() > 1e7 * Dd.min():
                    hpop.append(P.copy())
                    raise _RunStopped

            else:  # jSO --------------------------- [CODE] l.538-752
                FES0 = fes()
                Fpole = np.empty(N)
                CRpole = np.empty(N)
                pp = pmax - (pmax - pmin) * (FES0 / maxFES)
                use_eig = rng.random() < p_eig
                if use_eig:
                    o = np.argsort(fit, kind="stable")
                    E = _eig_vectors(P[o], n_eig_rows(N))
                order = np.argsort(fit, kind="stable")
                Asize = A.shape[0]
                PA = np.vstack([P, A]) if Asize else P
                Q = np.empty((N, D))
                for i in range(N):
                    rr = int(rng.integers(Hm))
                    CR = MCR[rr] + math.sqrt(0.1) * rng.standard_normal()
                    CR = min(max(CR, 0.0), 1.0)
                    if FES0 < 0.25 * maxFES:
                        CR = max(CR, 0.7)
                    elif FES0 < 0.5 * maxFES:
                        CR = max(CR, 0.6)
                    F = -1.0
                    while F <= 0:
                        F = 0.1 * math.tan(rng.random() * math.pi - math.pi / 2) + MF[rr]
                    F = min(F, 1.0)
                    if FES0 < 0.6 * maxFES and F > 0.7:
                        F = 0.7
                    Fpole[i] = F
                    CRpole[i] = CR
                    p = max(2, math.ceil(pp * N))
                    xpbest = P[order[int(rng.integers(p))]]
                    xi = P[i]
                    r1 = _rand_excl(rng, N, 1, [i])[0]
                    r2 = _rand_excl(rng, N + Asize, 1, [i, r1])[0]
                    if FES0 < 0.2 * maxFES:
                        Fw = 0.7 * F
                    elif FES0 < 0.4 * maxFES:
                        Fw = 0.8 * F
                    else:
                        Fw = 1.2 * F
                    v = xi + Fw * (xpbest - xi) + F * (P[r1] - PA[r2])
                    if use_eig:
                        y = _eig_cross(rng, E, xi, v, CBCR[i])
                        if self.fix_eig_bounds:
                            y = _mirror(y, lo, hi)
                    else:
                        y = xi.copy()
                        change = rng.random(D) < CR
                        if not change.any():
                            change[int(rng.integers(D))] = True
                        y[change] = v[change]
                        y = _mirror(y, lo, hi)
                    Q[i] = y
                Qf = np.empty(N)
                for i in range(N):
                    Qf[i] = ev(Q[i])
                SCR, SF, dlt = [], [], []
                suc = 0
                for i in range(N):
                    if Qf[i] < fit[i]:
                        dlt.append(fit[i] - Qf[i])
                        suc += 1
                        if A.shape[0] < Asize_max:
                            A = np.vstack([A, P[i]])
                        else:
                            A[int(rng.integers(A.shape[0]))] = P[i]
                        SCR.append(CRpole[i])
                        SF.append(Fpole[i])
                    if Qf[i] <= fit[i]:
                        P[i] = Q[i]
                        fit[i] = Qf[i]
                if suc > 0:
                    SCRa, SFa, d = np.array(SCR), np.array(SF), np.array(dlt)
                    MCR_old, MF_old = MCR[k], MF[k]
                    sd = d.sum()
                    wv = d / sd if sd > 0 else np.full(d.size, 1.0 / d.size)
                    if MCR[k] == -1 or SCRa.max() == 0:
                        MCR[k] = -1
                    else:
                        MCR[k] = np.sum(wv * SCRa * SCRa) / np.sum(wv * SCRa)
                    MF[k] = np.sum(wv * SFa * SFa) / np.sum(wv * SFa)
                    MCR[k] = (MCR[k] + MCR_old) / 2
                    MF[k] = (MF[k] + MF_old) / 2
                    k += 1
                    if k >= Hm - 1:  # MATLAB: k=k+1; if k>=H, k=1 (1-based)
                        k = 0
                ni[hh] += suc

            hpop.append(P.copy())

            # --- LPSR ([CODE] l.766-794) -------------------------------
            optN = _mround(((Nmin - N_init) / maxFES) * fes() + N_init)
            if N > optN:
                diff = N - optN
                if N - diff < Nmin:
                    diff = N - Nmin
                N -= diff
                sort_P()
                P, fit, CBF, CBCR = P[:N], fit[:N], CBF[:N], CBCR[:N]
                Asize_max = _mround(N * 2.6)
                while A.shape[0] > Asize_max:
                    A = np.delete(A, int(rng.integers(A.shape[0])), axis=0)
                mu = int(math.floor(N / 2))
                weights = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / weights.sum()
                mueff = weights.sum() ** 2 / np.sum(weights ** 2)


class EA4eigJsoIdebdOptimizer(EA4eigOptimizer):
    """EA4eig restricted to IDEbd + jSO — rank 1 of [RB] Table 2 and the best
    method on BBOB 40-D in [RB] Table 11 ([RBCPP] ``EA4EIGjSO_IDE``).

    [RBCPP] keeps the original IDEbd r1 and t_max behaviour ("code does what
    the original version is doing"), so both fixes default to False here;
    the reset threshold becomes 1/(5*2) = 0.1 as in [RBCPP].
    """

    def __init__(self, benchmark: BenchmarkFunction, seed: int = 42, **params):
        params.setdefault("components", ("idebd", "jso"))
        params.setdefault("fix_idebd_r1", False)
        params.setdefault("fix_idebd_tmax", False)
        super().__init__(benchmark, seed, **params)


class EA4eigSimplifiedOptimizer(BaseOptimizer):
    """Biedrzycki's final simplification of EA4eig ([RB] Sec. 3.3-4,
    "IDEbd, corr. t_max, corr. r1 selection, simplified mut., no rand. feat.
    in x_r3"; [RBCPP] ``onlyIDEcorrItMaxWithoutSortsNoIDEbdmutNoRandxr3CleaningWithoutFails``).

    Only the IDEbd component remains, with L-SHADE population reduction and
    the CoBiDE-style Eigen crossover. Per generation (population sorted
    ascending):
      superior ratio ps = ps_min + (1-ps_min) 10^(ps_shape (t/t_max - 1));
      for each i: b = i while t <= t_th, else random; r1, r2, r3 random
      (distinct, != i); F ~ N(b/N, sigma_n) redrawn until in (0, 1];
      if b and r1 are both inferior, r1 is redrawn from the superior part
      excluding the indices already drawn;  v = x_b + F(x_r1 - x_b) + F(x_r2 - x_r3).
      With probability p_eig: Eigen crossover (eigenvectors of the best
      int(N*eig_pop_ratio) members; per-slot CR ~ Cauchy(0.1 w.p.
      small_cauchy_thres else 0.95, sigma_c) clipped to [0,1], drawn once),
      then reflection at the bounds; else binomial with CR ~ N(i/N, sigma_n),
      out-of-range coordinates re-drawn uniformly. Selection <=. t_max =
      maxFES/N_init, raised to t when t exceeds it; t_th = t_max / t_th_div.

    Parameters (defaults = [RB] Table 5 "default"; ``tuned=True`` switches all
    of them to the Table 5 "tuned" column, irace-tuned on CEC 2022):
      pop_size=100 (tuned 76), min_pop_size=10 (19), p_eig=0.4 (0.18),
      sigma_n=0.1 (0.26), eig_pop_ratio=0.5 (0.29), ps_shape=5 (4.14),
      t_th_div=2 (2.31), ps_min=0.1 (0.08), small_cauchy_thres=0.5 (0.49),
      sigma_c=0.1 (0.19). (fails_th div is irrelevant: that code was removed.)
      ps_int_division=False: [RBCPP] computes ``t/tMax`` with C++ integer
      division, so ps stays at ~ps_min until t reaches t_max. The paper's
      Fig. 3 formula is real-valued, which is the default here; True
      reproduces the C++ code that generated [RB]'s published numbers.

    Bound handling: reflection after Eigen crossover, uniform re-draw of the
    offending coordinate after binomial crossover ([RBCPP]).
    """

    _TUNED = dict(pop_size=76, min_pop_size=19, p_eig=0.18, sigma_n=0.26,
                  eig_pop_ratio=0.29, ps_shape=4.14, t_th_div=2.31,
                  ps_min=0.08, small_cauchy_thres=0.49, sigma_c=0.19)

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int = 100,
        min_pop_size: int = 10,
        p_eig: float = 0.4,
        sigma_n: float = 0.1,
        eig_pop_ratio: float = 0.5,
        ps_shape: float = 5.0,
        t_th_div: float = 2.0,
        ps_min: float = 0.1,
        small_cauchy_thres: float = 0.5,
        sigma_c: float = 0.1,
        ps_int_division: bool = False,
        tuned: bool = False,
    ):
        super().__init__(benchmark, seed)
        vals = dict(pop_size=pop_size, min_pop_size=min_pop_size, p_eig=p_eig,
                    sigma_n=sigma_n, eig_pop_ratio=eig_pop_ratio,
                    ps_shape=ps_shape, t_th_div=t_th_div, ps_min=ps_min,
                    small_cauchy_thres=small_cauchy_thres, sigma_c=sigma_c)
        if tuned:
            vals.update(self._TUNED)
        for kname, kval in vals.items():
            setattr(self, kname, kval)
        self.pop_size = int(self.pop_size)
        self.min_pop_size = int(min(self.min_pop_size, self.pop_size))
        self.ps_int_division = ps_int_division
        self.tuned = tuned

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        D = self.dim
        hx: list[np.ndarray] = []
        hf: list[float] = []
        hpop: list[np.ndarray] = []
        func = self.func
        maxFES = int(max_evals)

        def ev(x: np.ndarray) -> float:
            if len(hf) >= maxFES:
                raise _BudgetExhausted
            x = np.array(x, dtype=float)
            f = float(func(x))
            hx.append(x)
            hf.append(f)
            return f

        try:
            self._run(rng, ev, hf, hpop, lo, hi, D, maxFES)
        except _BudgetExhausted:
            pass
        return self._make_result(hx, hf, hpop)

    def _run(self, rng, ev, hf, hpop, lo, hi, D, maxFES):
        width = hi - lo
        N_init = self.pop_size
        N = N_init
        Nmin = self.min_pop_size
        sn = self.sigma_n
        P = lo + rng.random((N, D)) * width
        fit = np.empty(N)
        for i in range(N):
            fit[i] = ev(P[i])
        hpop.append(P.copy())
        tMax = max(1, maxFES // N)
        tTh = tMax / self.t_th_div
        t = 0
        CRt = np.empty(N)
        for i in range(N):
            c = (_cauchy(rng, 0.1, self.sigma_c) if rng.random() < self.small_cauchy_thres
                 else _cauchy(rng, 0.95, self.sigma_c))
            CRt[i] = min(max(c, 0.0), 1.0)

        while len(hf) < maxFES:
            if t > tMax:
                tMax = t
            o = np.argsort(fit, kind="stable")
            P, fit, CRt = P[o], fit[o], CRt[o]
            Q = P.copy()
            ratio = (t // tMax) if self.ps_int_division else (t / tMax)
            sr = self.ps_min + (1 - self.ps_min) * 10.0 ** (self.ps_shape * (ratio - 1.0))
            sup = int(sr * N)
            for i in range(N):
                sel = _rand_excl(rng, N, 4, [i])
                b = i if t <= tTh else sel[0]
                r1, r2, r3 = sel[1], sel[2], sel[3]
                Fo = (b + 1) / N + sn * rng.standard_normal()
                while Fo <= 0 or Fo > 1:
                    Fo = (b + 1) / N + sn * rng.standard_normal()
                if b + 1 > sup and r1 + 1 > sup:
                    cand = [c for c in range(sup) if c not in sel]
                    r1 = cand[int(rng.integers(len(cand)))] if cand else 0
                xb = P[b]
                Q[i] = xb + Fo * (P[r1] - xb) + Fo * (P[r2] - P[r3])
            if rng.random() < self.p_eig:
                E = _eig_vectors(P, int(N * self.eig_pop_ratio))
                for i in range(N):
                    Q[i] = _mirror(_eig_cross(rng, E, P[i], Q[i], CRt[i]), lo, hi)
            else:
                for i in range(N):
                    CR = (i + 1) / N + sn * rng.standard_normal()
                    while CR < 0 or CR > 1:
                        CR = (i + 1) / N + sn * rng.standard_normal()
                    jrand = int(rng.integers(D))
                    keep = ~(rng.random(D) <= CR)
                    keep[jrand] = False
                    Q[i, keep] = P[i, keep]
                    bad = (Q[i] < lo) | (Q[i] > hi)
                    if bad.any():
                        Q[i, bad] = lo + rng.random(int(bad.sum())) * width
            for i in range(N):
                fq = ev(Q[i])
                if fq <= fit[i]:
                    P[i] = Q[i]
                    fit[i] = fq
            t += 1
            hpop.append(P.copy())
            optN = _mround(((Nmin - N_init) / maxFES) * len(hf) + N_init)
            if N > optN:
                diff = N - optN
                if N - diff < Nmin:
                    diff = N - Nmin
                N -= diff
                o = np.argsort(fit, kind="stable")
                P, fit, CRt = P[o][:N], fit[o][:N], CRt[o][:N]
