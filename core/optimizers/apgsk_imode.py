"""APGSK-IMODE — adaptive gaining-sharing knowledge hybridised with IMODE.

A. W. Mohamed, A. A. Hadi, P. Agrawal, K. M. Sallam and A. K. Mohamed,
"Gaining-Sharing Knowledge Based Algorithm with Adaptive Parameters Hybrid
with IMODE Algorithm for Solving CEC 2021 Benchmark Problems",
IEEE CEC 2021, pp. 841-848 (top-ranked entry of the CEC 2021 bound-constrained
competition).

Ported from the authors' MATLAB code distributed by the organisers
(github.com/P-N-Suganthan/2021-SO-BCO, ``Codes-of-top-methods (1).zip`` ->
``APGSK_IMODE Code.rar``: ``APGSK_IMODE.m``, ``IMODE.m``, ``APGSK_fun.m``,
``Introd_Par.m``, ``Gained_Shared_{Junior,Senior}_R1R2R3.m``, ``gnR1R2.m``,
``han_boun.m``, ``boundConstraint.m``, ``updateArchive.m``). The CEC 2022
copy in github.com/zhuzil/APGSK_IMODE_FL is algorithmically identical
(only budget / optimum table / logging differ). IMODE is implemented here,
inside this module, exactly as the copy shipped with APGSK-IMODE (no
SQP local search, memory 15·D).

Structure (APGSK_IMODE.m):
  NP = 30·D split into PS1 = 3/4·NP (IMODE) and PS2 = 1/4·NP (APGSK).
  Each iteration: cy += 1.
    cy == CS+1 (51): Probs from normalised best qualities
        q = [min f1, min f2]; Probs = clip((1 - q/Σq)/Σ, 0.1, 0.9);
        the better one gets Probs = one-hot (none if tied).
    cy == 2·CS (100): the winner's first row replaces a row of the loser
        (IMODE best -> APGSK row PS2; else APGSK row 1 -> IMODE row PS1 when
        strictly inside the box), loser re-sorted; cy = 1, Probs = [1, 1].
    IMODE generation if rand <= Probs(1), then LPSR of PS1 (to 4).
    APGSK generation if rand <  Probs(2), then LPSR of PS2 (to 12).
  So both sub-populations evolve for 50 iterations, then only the better one
  for 49, then the best individual is passed over, and so on.

IMODE generation (IMODE.m): F ~ Cauchy(M_F, 0.1), CR ~ N(M_CR, 0.1) from
a memory of size 15·D; population sorted, CR values sorted ascending (best
individual gets the smallest CR). Three operators chosen per individual with
probabilities prob (init 1/3):
  op1 current-to-φbest/1 with archive: x + F(φ - x + x_r1 - x̃_r2), φ from top 25 %
  op2 current-to-φbest/1:              x + F(φ - x + x_r1 - x_r3),  φ from top 25 %
  op3 weighted-rand-to-φbest:          F·x_r1 + F(φ - x_r3),         φ from top 50 %
Bound handling picks one of three rules at random per generation
(han_boun.m). Crossover: binomial with prob 0.3, otherwise exponential.
prob_k = clip(Δ_k / ΣΔ, 0.1, 0.9) with Δ_k the mean relative improvement
max(0, f - f_new)/|f| of operator k; if any Δ_k is 0, prob = 1/3 each
(MATLAB ``if count_S ~= 0`` on a vector is true only when all are nonzero).
Memory slot: weighted Lehmer means; no success -> slot set to 0.2 / 0.2.

APGSK generation (APGSK_fun.m): four (KF, KR) settings
KF_pool = [0.1, 1.0, 0.5, 1.0], KR_pool = [0.2, 0.1, 0.9, 0.9], chosen per
individual with probabilities KW. During the first 10 % of the budget
KW = [0.85, 0.05, 0.05, 0.05]; afterwards KW <- 0.95·KW + 0.05·Imp
(Imp = normalised fitness improvement per setting, floor 0.05), and KF is
taken from the pool only when rand >= 0.1 and nfes > 50 % — otherwise
every KF is -0.1 (``KF_poool``; as in the code). Junior dimensions
D_J = round(D·(1 - g)^0.5) with prob 1 - g, else round(D·(1 - g)^2),
g = nfes / max_nfes (adaptive knowledge rate); senior = D - D_J.
Junior: rank neighbours (better / worse) + random R3; senior: top 10 %,
middle 80 %, bottom 10 %. Midpoint bound repair, greedy (strict) selection.

Parameters (default — source):
  pop_size_factor = 30        NP = 30·D                   — Introd_Par.m
  apgsk_share     = 0.25      PS2 = NP/4                  — APGSK_IMODE.m
  min_pop_imode   = 4                                     — APGSK_IMODE.m
  min_pop_apgsk   = 12                                    — APGSK_IMODE.m
  cycle           = 50        CS                          — Introd_Par.m
  memory_factor   = 15        H = 15·D                    — APGSK_IMODE.m
  kf_pool / kr_pool / kf_alt / kw_init                    — APGSK_fun.m
  binomial_prob   = 0.3                                   — IMODE.m

Quirks reproduced from the code:
- ``updateArchive`` writes ``archive.Pop`` while IMODE reads ``archive.pop``,
  so IMODE's archive is always empty; op1 therefore draws x̃_r2 from the
  population only. No archive is kept here.
- han_boun rule 2 sends both lower- and upper-bound violators to the lower
  bound (``2*x_L - x2`` is used for the upper side); rule 3 uses a single
  uniform scalar for all lower violators and another for all upper ones.
- MATLAB ``min``/``max`` ignore NaN (``np.fmin``/``np.fmax`` here), e.g. a
  0/0 quality ratio gives Probs = [0.9, 0.9] (no winner).

Deviations: the budget is checked per evaluation; the competition's early
stop at error <= 1e-8 is not applied (the optimum is unknown to the
optimiser); RNG is NumPy's, seeded from ``self.seed``; the 2021 code's
feasibility test for sharing uses [-100, 100], here ``self.bounds``.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class APGSKIMODEOptimizer(BaseOptimizer):
    """APGSK-IMODE (Mohamed et al., CEC 2021). See module docstring."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size_factor: int = 30,
        apgsk_share: float = 0.25,
        min_pop_imode: int = 4,
        min_pop_apgsk: int = 12,
        cycle: int = 50,
        memory_factor: int = 15,
        binomial_prob: float = 0.3,
        kf_pool: tuple = (0.1, 1.0, 0.5, 1.0),
        kr_pool: tuple = (0.2, 0.1, 0.9, 0.9),
        kf_alt: float = -0.1,
        kw_init: tuple = (0.85, 0.05, 0.05, 0.05),
    ):
        super().__init__(benchmark, seed)
        self.pop_size_factor = pop_size_factor
        self.apgsk_share = apgsk_share
        self.min_pop_imode = min_pop_imode
        self.min_pop_apgsk = min_pop_apgsk
        self.cycle = cycle
        self.memory_factor = memory_factor
        self.binomial_prob = binomial_prob
        self.kf_pool = np.asarray(kf_pool, dtype=float)
        self.kr_pool = np.asarray(kr_pool, dtype=float)
        self.kf_alt = kf_alt
        self.kw_init = np.asarray(kw_init, dtype=float)
        # Diagnostics: (nfes, ran_imode, ran_apgsk) per iteration
        self.phase_log: list[tuple[int, int, int]] = []

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        with np.errstate(all="ignore"):
            return self._optimize(max_evals)

    # ------------------------------------------------------------------ #
    def _optimize(self, max_evals: int) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        D = self.dim
        self.phase_log = []

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
        pop = lo + rng.random((NP, D)) * (hi - lo)
        fit = evaluate(pop)
        if len(fit) < NP:
            return self._make_result(history_x, history_f, [pop[:len(fit)].copy()])
        history_pop.append(pop.copy())

        PS2 = int(round_half_away(NP * self.apgsk_share))
        PS1 = NP - PS2
        max_PS1, max_PS2 = PS1, PS2
        X1, f1 = pop[:PS1].copy(), fit[:PS1].copy()
        X2, f2 = pop[PS1:].copy(), fit[PS1:].copy()

        # IMODE state
        H = int(self.memory_factor * D)
        m_f = np.full(H, 0.5)
        m_cr = np.full(H, 0.5)
        hist_pos = 0
        prob = np.full(3, 1.0 / 3.0)
        rng.normal(0.5, 0.15, (2, NP))  # unused F / cr draws of the code
        # APGSK state
        kw = self.kw_init.copy()
        all_imp = np.zeros(4)

        probs = np.array([1.0, 1.0])
        indx = 0
        cy = 0
        CS = self.cycle
        done = False
        while not done and len(history_f) < max_evals:
            cy += 1
            if cy == int(np.ceil(CS + 1)):
                qual = np.array([f1.min(), f2.min()])
                nq = 1.0 - qual / qual.sum()
                probs = np.fmax(0.1, np.fmin(0.9, nq / nq.sum()))
                indx = int(np.argmax(probs)) + 1 if not np.isnan(probs).all() else 0
                if probs[0] == probs[1]:
                    indx = 0
                if indx > 0:
                    probs = np.zeros(2)
                    probs[indx - 1] = 1.0
            elif cy == 2 * int(np.ceil(CS)):
                if indx == 1:
                    X2[-1] = X1[0]
                    f2[-1] = f1[0]
                    o = np.argsort(f2, kind="stable")
                    X2, f2 = X2[o], f2[o]
                else:
                    if X2[0].min() > lo and X2[0].max() < hi:
                        X1[-1] = X2[0]
                        f1[-1] = f2[0]
                        o = np.argsort(f1, kind="stable")
                        X1, f1 = X1[o], f1[o]
                cy = 1
                probs = np.array([1.0, 1.0])

            ran1 = ran2 = 0
            if rng.random() <= probs[0]:
                ran1 = 1
                X1, f1, prob, m_f, m_cr, hist_pos, done = self._imode_gen(
                    rng, X1, f1, prob, m_f, m_cr, hist_pos, H, lo, hi, evaluate)
                if not done:
                    nfes = len(history_f)
                    plan = int(round_half_away((self.min_pop_imode - max_PS1) / max_evals
                                               * nfes + max_PS1))
                    X1, f1 = self._reduce(X1, f1, plan, self.min_pop_imode)
            if not done and rng.random() < probs[1]:
                ran2 = 1
                X2, f2, kw, all_imp, done = self._apgsk_gen(
                    rng, X2, f2, kw, all_imp, len(history_f), max_evals, lo, hi, evaluate)
                if not done:
                    nfes = len(history_f)
                    plan = int(round_half_away((self.min_pop_apgsk - max_PS2) / max_evals
                                               * nfes + max_PS2))
                    X2, f2 = self._reduce(X2, f2, plan, self.min_pop_apgsk)
            self.phase_log.append((len(history_f), ran1, ran2))
            history_pop.append(np.vstack([X1, X2]))

        return self._make_result(history_x, history_f, history_pop)

    @staticmethod
    def _reduce(X, f, plan, min_n):
        n = len(X)
        if n > plan:
            n_red = n - plan
            if n - n_red < min_n:
                n_red = n - min_n
            if n_red > 0:
                # MATLAB removes the worst one at a time (last of a stable sort)
                for _ in range(n_red):
                    worst = int(np.argsort(f, kind="stable")[-1])
                    X = np.delete(X, worst, axis=0)
                    f = np.delete(f, worst)
        return X, f

    # ------------------------------------------------------------------ #
    def _bound(self, rng, V, parent, lo, hi):
        """han_boun.m: one of three rules, chosen once per generation."""
        hb = int(rng.integers(1, 4))
        if hb == 1:
            low = V < lo
            V[low] = (parent[low] + lo) / 2
            high = V > hi
            V[high] = (parent[high] + hi) / 2
        elif hb == 2:
            low = V < lo
            V[low] = np.minimum(hi, np.maximum(lo, 2 * lo - parent[low]))
            high = V > hi
            V[high] = np.maximum(lo, np.minimum(hi, 2 * lo - parent[high]))
        else:
            low = V < lo
            V[low] = lo + rng.random() * (hi - lo)
            high = V > hi
            V[high] = lo + rng.random() * (hi - lo)
        return V

    def _imode_gen(self, rng, x, fitx, prob, m_f, m_cr, hist_pos, H, lo, hi, evaluate):
        n, D = x.shape
        idx = rng.integers(0, H, n)
        mu_sf, mu_cr = m_f[idx], m_cr[idx]
        cr = rng.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0.0
        cr = np.fmax(np.fmin(cr, 1.0), 0.0)
        F = mu_sf + 0.1 * np.tan(np.pi * (rng.random(n) - 0.5))
        bad = F <= 0
        while bad.any():
            F[bad] = mu_sf[bad] + 0.1 * np.tan(np.pi * (rng.random(int(bad.sum())) - 0.5))
            bad = F <= 0
        F = np.fmin(F, 1.0)
        o = np.argsort(fitx, kind="stable")
        fitx, x = fitx[o], x[o]
        cr = np.sort(cr)

        # gnR1R2: r1 != r0; r2 != r0, r1; r3 != r0, r1, r2 (archive empty)
        r0 = np.arange(n)
        r1 = rng.integers(0, n, n)
        while (b := r1 == r0).any():
            r1[b] = rng.integers(0, n, int(b.sum()))
        r2 = rng.integers(0, n, n)
        while (b := (r2 == r1) | (r2 == r0)).any():
            r2[b] = rng.integers(0, n, int(b.sum()))
        r3 = rng.integers(0, n, n)
        while (b := (r3 == r0) | (r3 == r1) | (r3 == r2)).any():
            r3[b] = rng.integers(0, n, int(b.sum()))

        bb = rng.random(n)
        l2 = prob[0] + prob[1]
        op1 = bb <= prob[0]
        op2 = (bb > prob[0]) & (bb <= l2)
        op3 = (bb > l2) & (bb <= 1.0)

        vi = np.zeros((n, D))
        pNP = max(int(round_half_away(0.25 * n)), 1)
        phix = x[rng.integers(0, pNP, n)]
        Fc = F[:, None]
        vi[op1] = x[op1] + Fc[op1] * (phix[op1] - x[op1] + x[r1[op1]] - x[r2[op1]])
        vi[op2] = x[op2] + Fc[op2] * (phix[op2] - x[op2] + x[r1[op2]] - x[r3[op2]])
        pNP = max(int(round_half_away(0.5 * n)), 2)
        phix = x[rng.integers(0, pNP, n)]
        vi[op3] = Fc[op3] * x[r1[op3]] + Fc[op3] * (phix[op3] - x[r3[op3]])

        vi = self._bound(rng, vi, x, lo, hi)
        if rng.random() < self.binomial_prob:
            mask = rng.random((n, D)) > cr[:, None]
            mask[r0, rng.integers(0, D, n)] = False
            ui = np.where(mask, x, vi)
        else:
            ui = x.copy()
            start = rng.integers(0, D, n)
            for i in range(n):
                l = start[i]
                while rng.random() < cr[i] and l < D - 1:
                    l += 1
                ui[i, start[i]:l + 1] = vi[i, start[i]:l + 1]

        fnew = evaluate(ui)
        if len(fnew) < n:  # budget exhausted mid-generation
            k = len(fnew)
            imp = fnew < fitx[:k]
            x[:k][imp] = ui[:k][imp]
            fitx[:k][imp] = fnew[imp]
            return x, fitx, prob, m_f, m_cr, hist_pos, True

        diff = np.abs(fitx - fnew)
        I = fnew < fitx
        good_cr, good_f = cr[I], F[I]

        diff2 = np.maximum(0.0, fitx - fnew) / np.abs(fitx)
        count_s = np.zeros(3)
        for k, op in enumerate((op1, op2, op3)):
            count_s[k] = np.fmax(0.0, diff2[op].mean()) if op.any() else 0.0
        if np.all(count_s != 0):
            prob = np.fmax(0.1, np.fmin(0.9, count_s / count_s.sum()))
        else:
            prob = np.full(3, 1.0 / 3.0)

        fitx = np.where(I, fnew, fitx)
        x = np.where(I[:, None], ui, x)

        m_f = m_f.copy()
        m_cr = m_cr.copy()
        if len(good_cr) > 0:
            w = diff[I] / diff[I].sum()
            m_f[hist_pos] = np.dot(w, good_f ** 2) / np.dot(w, good_f)
            if good_cr.max() == 0 or m_cr[hist_pos] == -1:
                m_cr[hist_pos] = -1
            else:
                m_cr[hist_pos] = np.dot(w, good_cr ** 2) / np.dot(w, good_cr)
            hist_pos = (hist_pos + 1) % H
        else:
            m_cr[hist_pos] = 0.2
            m_f[hist_pos] = 0.2

        o = np.argsort(fitx, kind="stable")
        return x[o], fitx[o], prob, m_f, m_cr, hist_pos, False

    def _apgsk_gen(self, rng, pop, fitness, kw, all_imp, nfes, max_nfes, lo, hi, evaluate):
        n, D = pop.shape

        def draw_k(w):
            c = np.cumsum(w)
            r = rng.random(n)
            return np.minimum(np.searchsorted(c, r, side="left"), 3)

        if nfes < 0.1 * max_nfes:
            kw = self.kw_init.copy()
            kind = draw_k(kw)
            KF = self.kf_pool[kind]
            KR = self.kr_pool[kind]
        else:
            kw = 0.95 * kw + 0.05 * all_imp
            kw = kw / kw.sum()
            kind = draw_k(kw)
            KR = self.kr_pool[kind]
            if rng.random() >= 0.1 and nfes > 0.5 * max_nfes:
                KF = self.kf_pool[kind]
            else:
                KF = np.full(n, self.kf_alt)

        g = nfes / max_nfes
        if rng.random() > g:
            dj = int(np.ceil(round_half_away(D * (1 - g) ** 0.5)))
        else:
            dj = int(np.ceil(round_half_away(D * (1 - g) ** 2)))

        ind_best = np.argsort(fitness, kind="stable")
        rank = np.empty(n, dtype=int)
        rank[ind_best] = np.arange(n)
        # Junior: nearest better / worse in rank
        Rg1 = np.empty(n, dtype=int)
        Rg2 = np.empty(n, dtype=int)
        for i in range(n):
            r = rank[i]
            if r == 0:
                Rg1[i], Rg2[i] = ind_best[1], ind_best[2]
            elif r == n - 1:
                Rg1[i], Rg2[i] = ind_best[n - 3], ind_best[n - 2]
            else:
                Rg1[i], Rg2[i] = ind_best[r - 1], ind_best[r + 1]
        R0 = np.arange(n)
        Rg3 = rng.integers(0, n, n)
        while (b := (Rg3 == Rg2) | (Rg3 == Rg1) | (Rg3 == R0)).any():
            Rg3[b] = rng.integers(0, n, int(b.sum()))
        # Senior: top 10 % / middle 80 % / bottom 10 %
        t1 = int(round_half_away(n * 0.1))
        t9 = int(round_half_away(n * 0.9))
        top, mid, bot = ind_best[:t1], ind_best[t1:t9], ind_best[t9:]
        R1 = top[rng.integers(0, len(top), n)]
        R2 = mid[rng.integers(0, len(mid), n)]
        R3 = bot[rng.integers(0, len(bot), n)]

        KFc = KF[:, None]
        junior = np.empty((n, D))
        a = fitness > fitness[Rg3]
        junior[a] = pop[a] + KFc[a] * (pop[Rg1[a]] - pop[Rg2[a]] + pop[Rg3[a]] - pop[a])
        a = ~a
        junior[a] = pop[a] + KFc[a] * (pop[Rg1[a]] - pop[Rg2[a]] + pop[a] - pop[Rg3[a]])
        senior = np.empty((n, D))
        a = fitness > fitness[R2]
        senior[a] = pop[a] + KFc[a] * (pop[R1[a]] - pop[a] + pop[R2[a]] - pop[R3[a]])
        a = ~a
        senior[a] = pop[a] + KFc[a] * (pop[R1[a]] - pop[R2[a]] + pop[a] - pop[R3[a]])
        for V in (junior, senior):
            low = V < lo
            V[low] = (pop[low] + lo) / 2
            high = V > hi
            V[high] = (pop[high] + hi) / 2

        jmask = rng.random((n, D)) <= dj / D
        smask = ~jmask
        jmask &= rng.random((n, D)) <= KR[:, None]
        smask &= rng.random((n, D)) <= KR[:, None]
        ui = pop.copy()
        ui[jmask] = junior[jmask]
        ui[smask] = senior[smask]

        child = evaluate(ui)
        if len(child) < n:
            k = len(child)
            imp = child < fitness[:k]
            pop[:k][imp] = ui[:k][imp]
            fitness[:k][imp] = child[imp]
            return pop, fitness, kw, all_imp, True

        dif = np.abs(fitness - child)
        better = fitness > child
        imp_k = np.array([dif[better & (kind == k)].sum() for k in range(4)])
        if imp_k.sum() != 0:
            imp_k = imp_k / imp_k.sum()
            order = np.argsort(imp_k, kind="stable")
            for j in order[:-1]:
                imp_k[j] = max(imp_k[j], 0.05)
            imp_k[order[-1]] = 1 - imp_k[order[:-1]].sum()
        else:
            imp_k = np.full(4, 0.25)
        pop = np.where(better[:, None], ui, pop)
        fitness = np.where(better, child, fitness)
        return pop, fitness, kw, imp_k, False


def round_half_away(v: float) -> float:
    """MATLAB round (half away from zero)."""
    return float(np.sign(v) * np.floor(np.abs(v) + 0.5))
