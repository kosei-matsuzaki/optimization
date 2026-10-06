"""UMOEAs-II — United Multi-Operator EAs II (CEC 2016 single-objective winner).

Source: S. Elsayed, N. Hamza, R. Sarker, "Testing united multi-operator
evolutionary algorithms-II on single objective optimization problems",
IEEE CEC 2016, pp. 2966-2973, doi:10.1109/CEC.2016.7744164
(open copy: http://hdl.handle.net/1959.4/unsworks_52367). Algorithm 1,
Eqs. (8)-(25), Sec. IV parameters.

Implemented as stated in the paper:
  * Two sub-populations: X1 evolved by MODE, X2 by CMA-ES. Per generation, MODE
    runs with probability prob1 and CMA-ES with prob2 (both 1 at first).
  * Cycle of CS generations (CS = 50 for D <= 10, 100 for 30-D, 150 above).
    At cy = CS prob1/prob2 are set from the improvement index
    I_i = (1 - NQ_i) + Ndiv_i, NQ_i = f_best_i / (f_best_1 + f_best_2),
    Ndiv_i = div_i / (div_1 + div_2), div_i = sum of distances to the
    sub-population best, prob_i = max(0.1, min(0.9, I_i / sum I)); both 1 if
    sum I = 0 (Eqs. 21-25).
  * At cy = 2*CS, information sharing then prob1 = prob2 = 1, cy = 0: if
    prob1 > prob2, X2 <- PS2 random members of X1 and CMA-ES is reset with
    sigma = sigma_init * (1 - cfe / FFEmax) (mean = arithmetic mean of X2,
    Sec. III-C); otherwise the worst of X1 is replaced by the best of X2.
  * MODE: DE1 = current-to-pbest/1 with archive (Eq. 8), DE2 = current-to-pbest/1
    without archive (Eq. 9), DE3 = weighted-rand-to-phi-best (Eq. 10, lambda=1,
    phi from the best 50 %), binomial crossover; p-best from the best 10 %.
    Operator probabilities from improvement rates IDE_i (Eq. 11/12), clipped to
    [0.1, 0.9]. SHADE memory (H = 6) for F/Cr (Eqs. 14-20). Archive 2.6*PS1.
    Linear population size reduction 18D -> 4 (Eq. 13).
  * CMA-ES: PS2 = 4 + floor(3 ln D), mu = PS2/2, sigma = 0.3 (see guess below).
  * Local search: in the last 25 % of the budget, each generation with
    probability prob_ls (0.1; set to 1e-4 after a failure, back to 0.1 after a
    success) the interior-point method is applied to the best solution for up to
    0.2 * FFEmax evaluations; on success the worst of X1 is replaced and the
    CMA-ES mean is moved to the new point with a small sigma.

Guesses (not fixed by the paper):
  * sigma_init = 0.3 * (hi - lo) (paper: "sigma = 0.3", scale unstated).
  * Interior-point method = scipy ``trust-constr`` (barrier/interior-point for
    the bound constraints, keep_feasible) with 2-point finite differences; the
    paper used MATLAB's fmincon interior-point.
  * "small sigma" after a successful LS = 1e-3 * (hi - lo).
  * IDE_i denominator uses |f_old| (Eq. 11 assumes non-negative errors); the
    probabilities are reset to 1/3 when every IDE_i is 0. Probabilities are
    renormalised before sampling an operator.
  * NQ uses errors shifted to be non-negative when a raw f is negative.
  * Information-sharing tie (prob1 == prob2) is treated as "CMA-ES best".
  * If pycma signals convergence, CMA-ES is re-initialised at the best of X2
    with sigma_init * (1 - cfe/FFEmax) (the paper has no CMA-ES restart rule
    besides information sharing).
  * F ~ Cauchy(M_F, 0.1) regenerated while <= 0 and capped at 1; Cr ~ N(M_Cr,
    0.1) clipped to [0, 1]; bound repair = midpoint to the violated bound
    (as in L-SHADE). Population reduction drops the worst members.
  * FFEmax = max_evals (the paper used 10000*D).
"""
from __future__ import annotations

import math

import numpy as np
import cma

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    pass


_NON_RESTART_KEYS = {"maxfevals", "maxiter", "ftarget"}


class UMOEAIIOptimizer(BaseOptimizer):
    """UMOEAs-II: MODE + CMA-ES with adaptive emphasis and interior-point LS."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        ps1_max_per_dim: int = 18,  # Sec. IV: PS1,max = 18D
        ps1_min: int = 4,           # Sec. IV: PS1,min = 4
        memory_size: int = 6,       # Sec. IV: H = 6
        archive_rate: float = 2.6,  # Sec. III-B: archive size 2.6 * PS1
        p_best: float = 0.1,        # Sec. III-B: x_p from the best 10 %
        phi_best: float = 0.5,      # Sec. III-B: x_phi from the best 50 %
        cycle: int | None = None,   # Sec. IV: CS = 50 (10D) / 100 (30D) / 150
        sigma_frac: float = 0.3,    # Sec. IV "sigma = 0.3"; scale = guess
        prob_ls: float = 0.1,       # Sec. III-F
        prob_ls_fail: float = 1e-4,  # Sec. III-F ("0001", read as 0.0001)
        ls_budget_frac: float = 0.2,  # Sec. IV: cfe_LS = 0.2 * FFEmax
        ls_start_frac: float = 0.75,  # Alg. 1 l.20: last 25 % of the budget
        ls_sigma_frac: float = 1e-3,  # guess: "small" sigma after LS success
    ):
        super().__init__(benchmark, seed)
        D = benchmark.dim
        self.ps1_max = ps1_max_per_dim * D
        self.ps1_min = ps1_min
        self.ps2 = 4 + int(3 * math.log(D))
        self.H = memory_size
        self.archive_rate = archive_rate
        self.p_best = p_best
        self.phi_best = phi_best
        self.cycle = cycle if cycle is not None else (
            50 if D <= 10 else (100 if D <= 30 else 150))
        self.sigma_frac = sigma_frac
        self.prob_ls0 = prob_ls
        self.prob_ls_fail = prob_ls_fail
        self.ls_budget_frac = ls_budget_frac
        self.ls_start_frac = ls_start_frac
        self.ls_sigma_frac = ls_sigma_frac
        # diagnostics: (eval_count, [prob1, prob2]) at every cy = CS update,
        # and (eval_count, [probDE1..3]) per MODE generation
        self.prob_history: list[tuple[int, list[float]]] = []
        self.op_prob_history: list[tuple[int, list[float]]] = []
        self.ls_log: list[tuple[int, bool]] = []

    # ── evaluation bookkeeping ────────────────────────────────────────────
    def _eval(self, x: np.ndarray) -> float:
        if len(self._hf) >= self._max_evals:
            raise _BudgetExhausted()
        x = np.asarray(x, dtype=float).copy()
        f = float(self.func(x))
        self._hx.append(x)
        self._hf.append(f)
        return f

    def _new_cma(self, x0: np.ndarray, sigma: float):
        lo, hi = self.bounds
        self._cma_count += 1
        opts = cma.CMAOptions()
        opts["seed"] = int(self.seed) * 7919 + 1000 * self._cma_count + 3
        opts["bounds"] = [[lo] * self.dim, [hi] * self.dim]
        opts["popsize"] = self.ps2
        opts["CMA_mu"] = self.ps2 // 2
        opts["maxfevals"] = np.inf
        opts["verbose"] = -9
        return cma.CMAEvolutionStrategy(np.asarray(x0, float),
                                        max(float(sigma), 1e-12), opts)

    # ── MODE generation ───────────────────────────────────────────────────
    def _mode_generation(self, rng, st):
        lo, hi = self.bounds
        D = self.dim
        X1, f1 = st["X1"], st["f1"]
        n = len(X1)
        order = np.argsort(f1, kind="stable")
        X1, f1 = X1[order], f1[order]
        A = st["archive"]
        pool = np.vstack([X1, A]) if len(A) else X1
        probs = np.asarray(st["prob_de"]) / np.sum(st["prob_de"])
        n_p = max(1, int(round(self.p_best * n)))
        n_phi = max(1, int(round(self.phi_best * n)))

        newX, newf = X1.copy(), f1.copy()
        S_F, S_Cr, S_df = [], [], []
        num = np.zeros(3)
        den = np.zeros(3)
        failed_parents = []
        for z in range(n):
            op = int(rng.choice(3, p=probs))
            r = int(rng.integers(self.H))
            Cr = float(np.clip(rng.normal(st["M_Cr"][r], 0.1), 0.0, 1.0))
            F = 0.0
            while F <= 0.0:
                F = st["M_F"][r] + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
            F = min(F, 1.0)
            others = [k for k in range(n) if k != z]
            if len(others) < 3:
                others = list(range(n))
            r1, r3 = rng.choice(others, 2, replace=False)
            if op == 0:   # DE1: current-to-pbest/1 with archive (Eq. 8)
                xp = X1[rng.integers(n_p)]
                cand = [k for k in range(len(pool)) if k != z and k != r1]
                xr2 = pool[rng.choice(cand)] if cand else pool[rng.integers(len(pool))]
                v = X1[z] + F * (xp - X1[z] + X1[r1] - xr2)
            elif op == 1:  # DE2: current-to-pbest/1 without archive (Eq. 9)
                xp = X1[rng.integers(n_p)]
                v = X1[z] + F * (xp - X1[z] + X1[r1] - X1[r3])
            else:          # DE3: weighted-rand-to-phi-best (Eq. 10, lambda = 1)
                xphi = X1[rng.integers(n_phi)]
                v = F * X1[r1] + 1.0 * (xphi - X1[r3])
            mask = rng.random(D) <= Cr
            mask[rng.integers(D)] = True
            u = np.where(mask, v, X1[z])
            below, above = u < lo, u > hi
            u[below] = (X1[z][below] + lo) / 2.0
            u[above] = (X1[z][above] + hi) / 2.0
            fu = self._eval(u)
            num[op] += max(0.0, f1[z] - fu)
            den[op] += abs(f1[z])
            if fu <= f1[z]:
                if fu < f1[z]:
                    failed_parents.append(X1[z].copy())
                    S_F.append(F)
                    S_Cr.append(Cr)
                    S_df.append(abs(f1[z] - fu))
                newX[z], newf[z] = u, fu
        st["X1"], st["f1"] = newX, newf

        # archive of replaced parents, trimmed at random to 2.6 * PS1
        if failed_parents:
            A = np.vstack([A, np.array(failed_parents)]) if len(A) else np.array(failed_parents)
        # operator probabilities (Eqs. 11-12)
        ide = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        if ide.sum() > 0:
            st["prob_de"] = [float(v) for v in np.clip(ide / ide.sum(), 0.1, 0.9)]
        else:
            st["prob_de"] = [1 / 3] * 3
        # SHADE memory update (Eqs. 16-20)
        if S_F:
            w = np.array(S_df) / np.sum(S_df)
            SF, SC = np.array(S_F), np.array(S_Cr)
            k = st["mem_k"]
            st["M_Cr"][k] = float(np.sum(w * SC))
            st["M_F"][k] = float(np.sum(w * SF ** 2) / max(np.sum(w * SF), 1e-300))
            st["mem_k"] = (k + 1) % self.H
        # linear population size reduction (Eq. 13)
        target = int(round((self.ps1_min - self.ps1_max) / self._max_evals
                           * len(self._hf) + self.ps1_max))
        target = max(self.ps1_min, min(target, len(st["X1"])))
        order = np.argsort(st["f1"], kind="stable")[:target]
        st["X1"], st["f1"] = st["X1"][order], st["f1"][order]
        cap = int(round(self.archive_rate * target))
        if len(A) > cap:
            A = A[rng.choice(len(A), cap, replace=False)]
        st["archive"] = A if len(A) else np.empty((0, D))

    # ── CMA-ES generation ─────────────────────────────────────────────────
    def _cma_generation(self, st):
        es = st["es"]
        if set(es.stop().keys()) - _NON_RESTART_KEYS:
            b = int(np.argmin(st["f2"]))
            st["es"] = es = self._new_cma(st["X2"][b], self._sigma_now())
        X = es.ask()
        fX = [self._eval(x) for x in X]
        es.tell(X, fX)
        X, fX = np.asarray(X, float), np.asarray(fX)
        order = np.argsort(fX, kind="stable")
        st["X2"], st["f2"] = X[order], fX[order]

    def _sigma_now(self) -> float:
        lo, hi = self.bounds
        return self.sigma_frac * (hi - lo) * max(
            1.0 - len(self._hf) / self._max_evals, 1e-6)

    # ── interior-point local search ───────────────────────────────────────
    def _local_search(self, x0: np.ndarray, f0: float):
        from scipy.optimize import minimize, Bounds
        lo, hi = self.bounds
        cap = min(int(self.ls_budget_frac * self._max_evals),
                  self._max_evals - len(self._hf))
        if cap <= self.dim + 1:
            return None
        best = {"x": None, "f": f0, "n": 0}

        class _LSDone(Exception):
            pass

        def obj(x):
            if best["n"] >= cap:
                raise _LSDone()
            x = np.clip(np.asarray(x, float), lo, hi)
            f = self._eval(x)
            best["n"] += 1
            if f < best["f"]:
                best["x"], best["f"] = x.copy(), f
            return f

        try:
            minimize(obj, np.asarray(x0, float), method="trust-constr",
                     bounds=Bounds([lo] * self.dim, [hi] * self.dim,
                                   keep_feasible=True),
                     options={"maxiter": 10 ** 6, "gtol": 1e-12,
                              "xtol": 1e-14, "verbose": 0})
        except _LSDone:
            pass
        except _BudgetExhausted:
            raise
        except Exception:
            pass  # numerical failure in the LS: keep what was found
        return best

    # ── main loop (Algorithm 1) ───────────────────────────────────────────
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        import warnings
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        D = self.dim
        self._max_evals = max_evals
        self._hx: list[np.ndarray] = []
        self._hf: list[float] = []
        self._cma_count = 0
        self.prob_history = []
        self.op_prob_history = []
        self.ls_log = []
        history_pop: list[np.ndarray] = []
        sigma_init = self.sigma_frac * (hi - lo)

        st: dict = {}
        try:
            PS = self.ps1_max + self.ps2
            X = rng.uniform(lo, hi, (PS, D))
            fX = np.array([self._eval(x) for x in X])
        except _BudgetExhausted:
            return self._make_result(self._hx, self._hf, None)
        perm = rng.permutation(PS)
        i1, i2 = perm[:self.ps1_max], perm[self.ps1_max:]
        st.update(X1=X[i1], f1=fX[i1], X2=X[i2], f2=fX[i2],
                  archive=np.empty((0, D)), prob_de=[1 / 3] * 3,
                  M_F=[0.5] * self.H, M_Cr=[0.5] * self.H, mem_k=0)
        st["es"] = self._new_cma(st["X2"].mean(axis=0), sigma_init)
        history_pop.append(np.vstack([st["X1"], st["X2"]]))

        prob = [1.0, 1.0]
        prob_ls = self.prob_ls0
        cy = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                while len(self._hf) < max_evals:
                    cy += 1
                    if cy == self.cycle:
                        prob = self._update_probs(st)
                        self.prob_history.append((len(self._hf), list(prob)))
                    if cy == 2 * self.cycle:
                        self._share(rng, st, prob)
                        prob = [1.0, 1.0]
                        cy = 0
                    if rng.random() <= prob[0]:
                        self._mode_generation(rng, st)
                        self.op_prob_history.append(
                            (len(self._hf), list(st["prob_de"])))
                    if rng.random() <= prob[1]:
                        self._cma_generation(st)
                    history_pop.append(np.vstack([st["X1"], st["X2"]]))
                    if len(self._hf) >= self.ls_start_frac * max_evals \
                            and rng.random() <= prob_ls:
                        b = int(np.argmin(self._hf))
                        res = self._local_search(self._hx[b], self._hf[b])
                        ok = bool(res and res["x"] is not None)
                        self.ls_log.append((len(self._hf), ok))
                        if ok:
                            prob_ls = self.prob_ls0
                            w = int(np.argmax(st["f1"]))
                            st["X1"][w], st["f1"][w] = res["x"], res["f"]
                            o = np.argsort(st["f1"], kind="stable")
                            st["X1"], st["f1"] = st["X1"][o], st["f1"][o]
                            es = st["es"]
                            es.mean = es.gp.geno(
                                res["x"].copy(),
                                from_bounds=es.boundary_handler.inverse)
                            es.sigma = self.ls_sigma_frac * (hi - lo)
                        else:
                            prob_ls = self.prob_ls_fail
        except _BudgetExhausted:
            pass

        return self._make_result(self._hx, self._hf, history_pop)

    # ── cycle-level updates ───────────────────────────────────────────────
    def _update_probs(self, st) -> list[float]:
        fb = np.array([np.min(st["f1"]), np.min(st["f2"])], float)
        if fb.min() < 0:
            fb = fb - fb.min()
        nq = fb / fb.sum() if fb.sum() > 0 else np.full(2, 0.5)
        div = []
        for Xs, fs in ((st["X1"], st["f1"]), (st["X2"], st["f2"])):
            b = Xs[int(np.argmin(fs))]
            div.append(float(np.linalg.norm(Xs - b, axis=1).sum()))
        div = np.array(div)
        ndiv = div / div.sum() if div.sum() > 0 else np.full(2, 0.5)
        I = (1.0 - nq) + ndiv
        if I.sum() <= 0:
            return [1.0, 1.0]
        return [float(v) for v in np.clip(I / I.sum(), 0.1, 0.9)]

    def _share(self, rng, st, prob):
        if prob[0] > prob[1]:
            n1 = len(st["X1"])
            idx = rng.choice(n1, self.ps2, replace=n1 < self.ps2)
            st["X2"], st["f2"] = st["X1"][idx].copy(), st["f1"][idx].copy()
            o = np.argsort(st["f2"], kind="stable")
            st["X2"], st["f2"] = st["X2"][o], st["f2"][o]
            st["es"] = self._new_cma(st["X2"].mean(axis=0), self._sigma_now())
        else:
            w = int(np.argmax(st["f1"]))
            b = int(np.argmin(st["f2"]))
            st["X1"][w], st["f1"][w] = st["X2"][b].copy(), st["f2"][b]
