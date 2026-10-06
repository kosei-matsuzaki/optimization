"""AMALGAM-SO — self-adaptive multimethod search (Vrugt, Robinson & Hyman 2009).

Reference
---------
J. A. Vrugt, B. A. Robinson, J. M. Hyman, "Self-Adaptive Multimethod Search
for Global Optimization in Real-Parameter Spaces", IEEE Trans. Evol. Comput.
13(2):243-259, 2009.  Section / figure / table numbers below refer to it.

The original MATLAB code of the single-objective version was "available on
request" and is not public; the public ``jaspervrugt/AMALGAM`` toolbox is the
*multi-objective* version and was only used to confirm the boundary
reflection rule.  Everything the paper leaves open is marked ``GUESS``.

Outline (Fig. 1)
----------------
* One population P of size N (max-distance LHS initialisation, Fig. 2).
* Each generation, q methods (default CMA, GA, PSO — the configuration the
  paper carries forward, Sec. VI-A) all draw their N_i offspring from the
  *same* parents P; sum(N_i) = N.
* R = P ∪ Q (2N) is sorted and the next P is chosen by the species selection
  of Sec. III / Fig. 4 (distance thresholds alpha = 1e-10·δ ... 1e-1·δ, an
  equal number of members per threshold).  The best point is always kept
  (elitism).
* When a restart criterion of Sec. IV fires, N is doubled (Table III,
  pop_increase = 2) and {N_i} is recomputed from the reproductive success of
  the previous run (eq. 2), subject to minimum shares (Sec. V).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult

_ALL_METHODS = ("CMA", "GA", "PCX", "PSO", "DE")


class AMALGAMSOOptimizer(BaseOptimizer):
    """AMALGAM-SO (CMA-GA-PSO by default).

    Parameters (paper reference in brackets)
    ----------------------------------------
    methods : tuple of {"CMA","GA","PCX","PSO","DE"}
        Search operators run concurrently [Sec. II-B; CMA-GA-PSO recommended,
        Sec. VI-A; CMA-GA-DE reported as equally good].
    pop_size : int | None
        Initial population N [Sec. V: 10 / 15 / 20 at n = 10 / 30 / 50].
        None -> 10 for n <= 10, else round(10 + (n - 10) / 4)  (GUESS for
        dimensions the paper does not list; reproduces the three listed).
    pop_increase : float
        Population growth factor per restart [Table III: 2.0].
    cma_first_share : float
        N_CMA / N in the first run [Sec. V: 80 %]; the remainder is split
        equally among the other methods (GUESS: "user-defined").
    min_share_cma, min_share_other : float
        Minimum offspring shares [Sec. V: 25 % for CMA, 5 % for the others];
        rounded up so every method keeps at least one child.
    lhs_sample : int
        Size of the LHS pool S0 for the max-distance initialisation
        [Fig. 2: 10 000].
    sigma0_frac : float
        CMA initial step size as a fraction of (x_max - x_min)
        [Sec. II-B-1, Table III: 0.15].
    tol_fun : float      [Sec. IV: TolFun = 1e-5]
    tol_x_frac : float   [Sec. IV: TolX = 1e-9 · sigma0]
    max_cond : float     [Sec. IV crit. 5: 1e14]
    accept_rate : float  [Sec. IV GA crit. 3: 7.5e-2] — see GUESS below.
    ga_pc, ga_eta_m, ga_tournament
        GA crossover prob. [0.90], polynomial-mutation index [20]; mutation
        probability is 1/n [Sec. II-B-2, Table III].  Tournament pool size 4
        is a GUESS (paper: "a number of strings").
    pso_c1, pso_c2, pso_w_start, pso_w_end, pso_vmax_frac, pso_neigh_frac
        [Sec. II-B-4, Table III: c1 = c2 = 2, inertia 0.9 -> 0.4 linearly,
        v_max = 15 % of the range, neighbourhood best over the closest 10 %].
    de_F, de_CR     [Sec. II-B-5: DE/rand/1/bin, F = CR = 0.9]
    pcx_mu, pcx_sigma_zeta, pcx_sigma_eta  [Sec. II-B-3: 3, 0.1, 0.1
        (variances 0.01)]
    inject_rel_tol : float
        [Sec. II-A: the best point of the previous run replaces a random LHS
        member only if it improved by >= 1e-3 in relative terms.]

    Further GUESSes (paper silent or ambiguous)
    -------------------------------------------
    * CMA update in a shared population: the CMA mean / covariance / step
      size are updated from the best mu_CMA members of the *selected*
      population P_{t+1} (Sec. II-B-1: "the best mu_CMA individuals are
      selected as parental vectors"), so CMA learns from all methods' points.
      Steps are clipped in Mahalanobis norm to sqrt(n) + 2n/(n+2) (Hansen's
      injection rule) to keep CSA stable when foreign points are selected.
      lambda_CMA = N_CMA, mu_CMA = floor(lambda/2), weights
      ln(mu+1) - ln(k) [Table III]; other CMA constants are the cmaes.m
      defaults [ref. 21].
    * CMA boundary handling: the paper uses the cmaes.m v2.35 penalty; here
      CMA offspring are clipped to the box (the CMA update then uses the
      repaired points, so a penalty would never be seen by the update).
      Other methods reflect off the bound by the violation, then resample
      uniformly if still outside [Sec. V, and the public AMALGAM toolbox].
    * GA/PSO/DE stopping criteria (Sec. IV) are read as statistics of the
      best-so-far trajectory over a window of W = 10 + ceil(30 n / N)
      generations: (1) std of x_best < TolX in all coordinates,
      (2) range/max|f_best| < TolFun, (3) fraction of generations improving
      f_best < accept_rate (the paper's "·lambda" factor is dropped),
      (4) f_best unchanged for 50 + floor(100 n / N) generations.  Any one
      CMA or GA-group criterion triggers a restart.
    * Species selection: when a threshold level cannot be filled, the
      remaining slots go to the best unselected points; the quota remainder
      (N mod 10) goes to the smallest thresholds.
    * Eq. (2): Psi_i = summed decrease of the best-so-far value caused by
      offspring of method i during the run.  If no method improved, the
      previous shares are kept.  After applying the minima, the rounding
      residual goes to (or comes from) the method with the largest share.
    * PSO in a shared population: each member carries a velocity and a
      personal best; members created by other methods start with zero
      velocity and themselves as personal best.  Inertia decreases over the
      generations of the current run (t / t_max, t_max = remaining evals / N).
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        methods: Sequence[str] = ("CMA", "GA", "PSO"),
        pop_size: int | None = None,
        pop_increase: float = 2.0,
        cma_first_share: float = 0.80,
        min_share_cma: float = 0.25,
        min_share_other: float = 0.05,
        lhs_sample: int = 10000,
        sigma0_frac: float = 0.15,
        tol_fun: float = 1e-5,
        tol_x_frac: float = 1e-9,
        max_cond: float = 1e14,
        accept_rate: float = 7.5e-2,
        ga_pc: float = 0.90,
        ga_eta_m: float = 20.0,
        ga_tournament: int = 4,
        pso_c1: float = 2.0,
        pso_c2: float = 2.0,
        pso_w_start: float = 0.9,
        pso_w_end: float = 0.4,
        pso_vmax_frac: float = 0.15,
        pso_neigh_frac: float = 0.10,
        de_F: float = 0.9,
        de_CR: float = 0.9,
        pcx_mu: int = 3,
        pcx_sigma_zeta: float = 0.1,
        pcx_sigma_eta: float = 0.1,
        inject_rel_tol: float = 1e-3,
    ):
        super().__init__(benchmark, seed)
        methods = tuple(str(m).upper() for m in methods)
        bad = [m for m in methods if m not in _ALL_METHODS]
        if bad or not methods or len(set(methods)) != len(methods):
            raise ValueError(f"methods must be distinct members of {_ALL_METHODS}")
        self.methods = methods
        n = self.dim
        if pop_size is None:
            pop_size = 10 if n <= 10 else int(round(10 + (n - 10) / 4))
        self.pop_size = int(max(pop_size, len(methods) + 1))
        self.pop_increase = pop_increase
        self.cma_first_share = cma_first_share
        self.min_share_cma = min_share_cma
        self.min_share_other = min_share_other
        self.lhs_sample = lhs_sample
        self.sigma0_frac = sigma0_frac
        self.tol_fun = tol_fun
        self.tol_x_frac = tol_x_frac
        self.max_cond = max_cond
        self.accept_rate = accept_rate
        self.ga_pc = ga_pc
        self.ga_eta_m = ga_eta_m
        self.ga_tournament = ga_tournament
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_w_start = pso_w_start
        self.pso_w_end = pso_w_end
        self.pso_vmax_frac = pso_vmax_frac
        self.pso_neigh_frac = pso_neigh_frac
        self.de_F = de_F
        self.de_CR = de_CR
        self.pcx_mu = pcx_mu
        self.pcx_sigma_zeta = pcx_sigma_zeta
        self.pcx_sigma_eta = pcx_sigma_eta
        self.inject_rel_tol = inject_rel_tol
        # One dict per run: {"start_eval", "N", "counts", "psi", "best_f", "reason"}
        self.run_log: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _min_count(self, m: str, N: int) -> int:
        share = self.min_share_cma if m == "CMA" else self.min_share_other
        return max(1, int(math.ceil(share * N)))

    def _initial_counts(self, N: int) -> dict[str, int]:
        q = len(self.methods)
        if q == 1:
            return {self.methods[0]: N}
        counts: dict[str, int] = {}
        if "CMA" in self.methods:
            others = [m for m in self.methods if m != "CMA"]
            n_cma = int(round(self.cma_first_share * N))
            n_cma = min(max(n_cma, self._min_count("CMA", N)), N - len(others))
            counts["CMA"] = n_cma
            rest = N - n_cma
            for i, m in enumerate(others):
                counts[m] = rest // len(others) + (1 if i < rest % len(others) else 0)
        else:
            for i, m in enumerate(self.methods):
                counts[m] = N // q + (1 if i < N % q else 0)
        for m in self.methods:  # make sure nobody starts below 1
            if counts[m] < 1:
                donor = max(counts, key=counts.get)
                counts[donor] -= 1
                counts[m] = 1
        return counts

    def _update_counts(self, N: int, psi: dict[str, float],
                       prev: dict[str, int]) -> dict[str, int]:
        """Eq. (2) with the minimum shares of Sec. V."""
        tot = sum(psi.values())
        if tot > 0:
            frac = {m: psi[m] / tot for m in self.methods}
        else:  # GUESS: no information -> keep previous shares
            s = sum(prev.values())
            frac = {m: prev[m] / s for m in self.methods}
        counts = {m: int(math.floor(N * frac[m])) for m in self.methods}
        for m in self.methods:
            counts[m] = max(counts[m], self._min_count(m, N))
        order = sorted(self.methods, key=lambda m: -frac[m])
        diff = N - sum(counts.values())
        while diff > 0:
            counts[order[0]] += 1
            diff -= 1
        while diff < 0:  # take from the largest methods that are above minimum
            for m in order:
                if diff == 0:
                    break
                if counts[m] > self._min_count(m, N):
                    counts[m] -= 1
                    diff += 1
            else:
                if all(counts[m] <= self._min_count(m, N) for m in order):
                    break
        return counts

    def _maxdist_lhs(self, rng, N: int) -> np.ndarray:
        """Fig. 2: greedy max-min-distance subset of a large LHS sample."""
        n = self.dim
        S = self.lhs_sample
        cut = (np.arange(S)[:, None] + rng.random((S, n))) / S
        for j in range(n):
            cut[:, j] = cut[rng.permutation(S), j]
        P = [0]
        mind = np.sqrt(((cut - cut[0]) ** 2).sum(1))
        while len(P) < N:
            r = int(np.argmax(mind))
            P.append(r)
            mind = np.minimum(mind, np.sqrt(((cut - cut[r]) ** 2).sum(1)))
        lo, hi = self.bounds
        return lo + (hi - lo) * cut[P]

    def _reflect(self, X: np.ndarray, rng) -> np.ndarray:
        lo, hi = self.bounds
        X = np.where(X < lo, 2 * lo - X, X)
        X = np.where(X > hi, 2 * hi - X, X)
        bad = (X < lo) | (X > hi)
        if bad.any():
            X[bad] = rng.uniform(lo, hi, int(bad.sum()))
        return X

    def _species_select(self, X: np.ndarray, F: np.ndarray, N: int) -> np.ndarray:
        """Sec. III / Fig. 4: returns indices into the combined set (sorted)."""
        lo, hi = self.bounds
        delta = hi - lo
        order = np.argsort(F, kind="stable")
        Xs = X[order]
        M = len(order)
        alphas = delta * 10.0 ** np.arange(-10, 0)  # 1e-10 δ ... 1e-1 δ
        L = len(alphas)
        quota = [N // L + (1 if j < N % L else 0) for j in range(L)]
        selected = np.zeros(M, dtype=bool)
        mind = np.full(M, np.inf)
        n_sel = 0
        for j, a in enumerate(alphas):
            # a selected point has mind == 0 <= a, so it is never re-picked
            for _ in range(quota[j]):
                if n_sel >= N:
                    break
                ok = np.flatnonzero(mind > a)
                if len(ok) == 0:
                    break
                c = int(ok[0])  # best (sorted) admissible candidate
                selected[c] = True
                mind = np.minimum(mind, np.sqrt(((Xs - Xs[c]) ** 2).sum(1)))
                n_sel += 1
        if n_sel < N:  # GUESS: fill with the best remaining points
            for c in range(M):
                if n_sel >= N:
                    break
                if not selected[c]:
                    selected[c] = True
                    n_sel += 1
        return order[np.flatnonzero(selected)]

    # ----------------------------------------------------------- operators
    def _gen_ga(self, rng, X, F, k):
        N, n = X.shape
        lo, hi = self.bounds
        T = min(self.ga_tournament, N)
        out = np.empty((k, n))
        pm = 1.0 / n
        eta = self.ga_eta_m
        for i in range(k):
            pool = rng.choice(N, size=T, replace=False)
            pool = pool[np.argsort(F[pool], kind="stable")]
            p1, p2 = X[pool[0]], X[pool[1 % T]]
            if rng.random() < self.ga_pc:
                mask = rng.random(n) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            # polynomial mutation (Deb & Goyal 1996, as in NSGA-II)
            for j in range(n):
                if rng.random() < pm:
                    y = child[j]
                    d1 = (y - lo) / (hi - lo)
                    d2 = (hi - y) / (hi - lo)
                    u = rng.random()
                    mp = 1.0 / (eta + 1.0)
                    if u <= 0.5:
                        xy = 1.0 - d1
                        val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
                        dq = val ** mp - 1.0
                    else:
                        xy = 1.0 - d2
                        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
                        dq = 1.0 - val ** mp
                    child[j] = y + dq * (hi - lo)
            out[i] = child
        return self._reflect(out, rng)

    def _gen_de(self, rng, X, F, k):
        N, n = X.shape
        out = np.empty((k, n))
        for i in range(k):
            t = int(rng.integers(N))
            cand = [c for c in range(N) if c != t]
            if len(cand) >= 3:
                r1, r2, r3 = rng.choice(cand, 3, replace=False)
            else:
                r1, r2, r3 = rng.choice(N, 3, replace=True)
            v = X[r1] + self.de_F * (X[r2] - X[r3])
            jr = int(rng.integers(n))
            mask = rng.random(n) <= self.de_CR
            mask[jr] = True
            out[i] = np.where(mask, v, X[t])
        return self._reflect(out, rng)

    def _gen_pcx(self, rng, X, F, k):
        N, n = X.shape
        best = int(np.argmin(F))
        out = np.empty((k, n))
        mu = min(self.pcx_mu, N)
        for i in range(k):
            others = rng.choice([c for c in range(N) if c != best],
                                size=mu - 1, replace=False) if N > 1 else []
            idx = [best] + list(others)
            g = X[idx].mean(0)
            d = X[best] - g
            dn = np.linalg.norm(d)
            if dn > 0:
                e = d / dn
                dists = []
                for o in others:
                    v = X[o] - X[best]
                    perp = v - (v @ e) * e
                    dists.append(np.linalg.norm(perp))
                Dbar = float(np.mean(dists)) if dists else 0.0
                z = rng.standard_normal(n) * self.pcx_sigma_eta * Dbar
                z -= (z @ e) * e
                child = X[best] + rng.standard_normal() * self.pcx_sigma_zeta * d + z
            else:  # GUESS: degenerate parents -> isotropic step of their spread
                spread = np.mean([np.linalg.norm(X[o] - X[best]) for o in others]) if len(others) else 0.0
                child = X[best] + rng.standard_normal(n) * self.pcx_sigma_eta * spread
            out[i] = child
        return self._reflect(out, rng)

    def _gen_pso(self, rng, X, F, V, PB, k, w):
        N, n = X.shape
        lo, hi = self.bounds
        vmax = self.pso_vmax_frac * (hi - lo)
        kn = max(1, int(math.ceil(self.pso_neigh_frac * N)))
        D = cdist(X, X)
        parents = rng.choice(N, size=k, replace=k > N)
        out = np.empty((k, n))
        vel = np.empty((k, n))
        for i, p in enumerate(parents):
            neigh = np.argsort(D[p], kind="stable")[: kn + 1]  # self + closest 10 %
            nb = neigh[np.argmin(F[neigh])]
            r1 = rng.random(n)
            r2 = rng.random(n)
            v = (w * V[p] + self.pso_c1 * r1 * (PB[p] - X[p])
                 + self.pso_c2 * r2 * (X[nb] - X[p]))
            v = np.clip(v, -vmax, vmax)
            vel[i] = v
            out[i] = X[p] + v
        return self._reflect(out, rng), vel, parents

    # ------------------------------------------------------------- main
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        n = self.dim
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []
        self.run_log = []
        restart_bests: list[np.ndarray] = []

        def evaluate(Xc: np.ndarray) -> np.ndarray:
            fs = np.full(len(Xc), np.inf)
            for i, x in enumerate(Xc):
                if len(history_f) >= max_evals:
                    break
                f = float(self.func(x))
                history_x.append(np.array(x, dtype=float))
                history_f.append(f)
                fs[i] = f
            return fs

        N = self.pop_size
        counts = self._initial_counts(N)
        gbest_x = None
        gbest_f = np.inf
        prev_run_best = np.inf  # best of the run before the last one
        sigma0 = self.sigma0_frac * (hi - lo)
        tol_x = self.tol_x_frac * sigma0
        run_idx = 0

        while len(history_f) < max_evals:
            # ---------------- initial population of this run
            X = self._maxdist_lhs(rng, N)
            if gbest_x is not None:
                improved = (prev_run_best == np.inf or
                            gbest_f < prev_run_best - self.inject_rel_tol * abs(prev_run_best))
                if improved:
                    X[int(rng.integers(N))] = gbest_x
            F = evaluate(X)
            if len(history_f) >= max_evals and not np.isfinite(F).all():
                keep = np.isfinite(F)
                X, F = X[keep], F[keep]
                if len(F):
                    history_pop.append(X.copy())
                break
            history_pop.append(X.copy())
            V = np.zeros_like(X)
            PB = X.copy()
            PBF = F.copy()
            log = {"start_eval": len(history_f) - N, "N": N, "counts": dict(counts),
                   "psi": {m: 0.0 for m in self.methods}, "best_f": None,
                   "reason": "budget"}
            psi = log["psi"]
            run_best = float(F.min())
            t_max = max(1, (max_evals - len(history_f)) // N)

            # ---------------- CMA state
            use_cma = "CMA" in self.methods
            if use_cma:
                lam = counts["CMA"]
                mu = max(1, lam // 2)
                wts = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                wts /= wts.sum()
                mueff = 1.0 / (wts ** 2).sum()
                cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                cs = (mueff + 2) / (n + mueff + 5)
                c1 = 2 / ((n + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
                damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
                chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
                cy = math.sqrt(n) + 2 * n / (n + 2)
                o = np.argsort(F, kind="stable")[:mu]
                mean = wts @ X[o] if len(o) == mu else X[o].mean(0)
                sigma = sigma0
                C = np.eye(n)
                B = np.eye(n)
                Dd = np.ones(n)
                invsqrtC = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                cma_hist_window = 10 + int(math.ceil(30 * n / lam))
            ga_window = 10 + int(math.ceil(30 * n / N))
            flat_window = 50 + int(math.floor(100 * n / N))
            best_traj_f: list[float] = []
            best_traj_x: list[np.ndarray] = []
            improved_flags: list[bool] = []
            t = 0
            stop_reason = None

            while len(history_f) < max_evals:
                w_in = self.pso_w_start - (self.pso_w_start - self.pso_w_end) * min(1.0, t / t_max)
                kids: list[np.ndarray] = []
                origin: list[str] = []
                kidV: list[np.ndarray] = []
                kidPB: list[np.ndarray] = []
                kidPBF: list[float] = []
                cma_kids = None
                for m in self.methods:
                    k = counts[m]
                    if k <= 0:
                        continue
                    if m == "CMA":
                        Z = rng.standard_normal((k, n))
                        Y = (Z * Dd) @ B.T
                        Q = np.clip(mean + sigma * Y, lo, hi)
                        cma_kids = (len(kids), k)
                        vz = np.zeros((k, n)); pb = Q; pbf = [np.inf] * k
                    elif m == "GA":
                        Q = self._gen_ga(rng, X, F, k)
                        vz = np.zeros((k, n)); pb = Q; pbf = [np.inf] * k
                    elif m == "DE":
                        Q = self._gen_de(rng, X, F, k)
                        vz = np.zeros((k, n)); pb = Q; pbf = [np.inf] * k
                    elif m == "PCX":
                        Q = self._gen_pcx(rng, X, F, k)
                        vz = np.zeros((k, n)); pb = Q; pbf = [np.inf] * k
                    else:  # PSO
                        Q, vz, par = self._gen_pso(rng, X, F, V, PB, k, w_in)
                        pb = PB[par]; pbf = list(PBF[par])
                    kids.extend(Q); origin.extend([m] * k)
                    kidV.extend(vz); kidPB.extend(pb); kidPBF.extend(pbf)
                Qx = np.array(kids)
                Qf = evaluate(Qx)
                valid = np.isfinite(Qf)
                Qx, Qf = Qx[valid], Qf[valid]
                origin = [o_ for o_, v_ in zip(origin, valid) if v_]
                kidV = np.array(kidV)[valid]
                kidPB = np.array(kidPB)[valid].copy()
                kidPBF = np.array(kidPBF)[valid].copy()
                # personal bests of children
                better = Qf < kidPBF
                kidPB[better] = Qx[better]
                kidPBF[better] = Qf[better]
                if cma_kids is not None:
                    cvals = Qf[[i for i, o_ in enumerate(origin) if o_ == "CMA"]]
                else:
                    cvals = np.array([])

                # reproductive success (eq. 2 input)
                if len(Qf):
                    ib = int(np.argmin(Qf))
                    if Qf[ib] < run_best:
                        psi[origin[ib]] += run_best - Qf[ib]
                        run_best = float(Qf[ib])

                # species selection on R = P ∪ Q
                RX = np.vstack([X, Qx]) if len(Qx) else X
                RF = np.concatenate([F, Qf])
                RV = np.vstack([V, kidV]) if len(Qx) else V
                RPB = np.vstack([PB, kidPB]) if len(Qx) else PB
                RPBF = np.concatenate([PBF, kidPBF])
                sel = self._species_select(RX, RF, N)
                X, F, V, PB, PBF = RX[sel], RF[sel], RV[sel], RPB[sel], RPBF[sel]
                history_pop.append(X.copy())
                t += 1
                if len(history_f) >= max_evals:
                    break

                # ---------------- CMA update from the selected population
                if use_cma:
                    o = np.argsort(F, kind="stable")[:mu]
                    Ysel = (X[o] - mean) / sigma
                    if len(o) < mu:
                        Ysel = np.vstack([Ysel, np.zeros((mu - len(o), n))])
                    mnorm = np.sqrt(((Ysel @ invsqrtC.T) ** 2).sum(1))
                    scale = np.minimum(1.0, cy / np.maximum(mnorm, 1e-300))
                    Ysel = Ysel * scale[:, None]
                    yw = wts @ Ysel
                    mean = mean + sigma * yw
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
                    hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * t))
                            / chiN < 1.4 + 2 / (n + 1))
                    pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
                    C = ((1 - c1 - cmu) * C
                         + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
                         + cmu * (Ysel.T * wts) @ Ysel)
                    sigma *= math.exp(min(1.0, (cs / damps) * (np.linalg.norm(ps) / chiN - 1)))
                    sigma = min(sigma, 2.0 * (hi - lo))
                    C = np.triu(C) + np.triu(C, 1).T
                    ev, B = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-300)
                    Dd = np.sqrt(ev)
                    invsqrtC = (B / Dd) @ B.T

                # ---------------- restart criteria (Sec. IV)
                ib = int(np.argmin(F))
                best_traj_f.append(float(F[ib]))
                best_traj_x.append(X[ib].copy())
                improved_flags.append(len(best_traj_f) > 1 and best_traj_f[-1] < best_traj_f[-2])
                if use_cma:
                    if len(best_traj_f) >= cma_hist_window:
                        win = best_traj_f[-cma_hist_window:]
                        if max(win) - min(win) == 0:
                            stop_reason = "cma_flat"
                    if stop_reason is None and len(cvals) > 1:
                        mx = np.max(np.abs(cvals))
                        if mx > 0 and (cvals.max() - cvals.min()) / mx < self.tol_fun:
                            stop_reason = "cma_tolfun"
                    if stop_reason is None and np.all(sigma * Dd < tol_x) and \
                            np.all(sigma * np.abs(pc) < tol_x):
                        stop_reason = "cma_tolx"
                    if stop_reason is None:
                        i_ax = t % n
                        if np.all(mean == mean + 0.1 * sigma * Dd[i_ax] * B[:, i_ax]):
                            stop_reason = "cma_noeffectaxis"
                    if stop_reason is None and np.any(mean == mean + 0.2 * sigma * np.sqrt(np.diag(C))):
                        stop_reason = "cma_noeffectcoord"
                    if stop_reason is None and ev.max() / ev.min() > self.max_cond:
                        stop_reason = "cma_cond"
                if stop_reason is None and len(best_traj_f) >= ga_window:
                    wx = np.array(best_traj_x[-ga_window:])
                    wf = np.array(best_traj_f[-ga_window:])
                    mx = np.max(np.abs(wf))
                    if np.all(wx.std(0) < tol_x):
                        stop_reason = "ga_tolx"
                    elif mx > 0 and (wf.max() - wf.min()) / mx < self.tol_fun:
                        stop_reason = "ga_tolfun"
                    elif np.mean(improved_flags[-ga_window:]) < self.accept_rate:
                        stop_reason = "ga_accept"
                if stop_reason is None and len(best_traj_f) >= flat_window:
                    win = best_traj_f[-flat_window:]
                    if max(win) - min(win) == 0:
                        stop_reason = "ga_flat"
                if stop_reason is not None:
                    break

            # ---------------- end of run: bookkeeping and restart
            ib = int(np.argmin(F)) if len(F) else None
            if ib is not None:
                restart_bests.append(X[ib].copy())
                if F[ib] < gbest_f:
                    prev_run_best = gbest_f
                    gbest_f = float(F[ib])
                    gbest_x = X[ib].copy()
                else:
                    prev_run_best = gbest_f
            log["best_f"] = gbest_f
            log["reason"] = stop_reason or "budget"
            log["generations"] = t
            self.run_log.append(log)
            run_idx += 1
            N = int(round(N * self.pop_increase))
            counts = self._update_counts(N, psi, counts)

        solutions = restart_bests[:-1] + (list(history_pop[-1]) if history_pop else [])
        return self._make_result(history_x, history_f, history_pop,
                                 solutions=solutions or None)
