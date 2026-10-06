"""IMODE — faithful numpy port of the authors' MATLAB code (CEC 2020 winner).

    K. M. Sallam, S. M. Elsayed, R. K. Chakrabortty and M. J. Ryan,
    "Improved Multi-operator Differential Evolution Algorithm for Solving
    Unconstrained Problems", IEEE CEC 2020, pp. 1-8.

Source ported: "E24365 Updated codes" in the competition repository
github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark (the code that
produced the published results). File / line references below are to that
archive (IMODE_main.m = M, IMODE.m = I, Introd_Par.m = P).

Replaces the mealpy wrapper (``lib_wrappers.IMODEOptimizer``) as the baseline.
That wrapper re-draws the operator assignment after evaluation (so the
improvement is credited to random operators), uses memory size 5 / init 0.5,
and has no SQP phase. Here the operator mask that produced a trial is the
one credited, and the SQP local search is included.

Parameters and their source
---------------------------
- N_init = 6·D² (P:38), N_min = 4 (P:41); linear population size reduction
  by evaluations, applied *before* each generation, dropping the last rows
  (the population is kept sorted by fitness, so these are the worst) (M:126-146)
- three operators (I:63-70), chosen per individual by one uniform draw against
  the cumulative operator probabilities (I:50-55), initial probs 1/3 (M:47):
    op1 current-to-φbest/1 with archive:  v = x + F(xφ − x + x_r1 − x̃_r2)
    op2 current-to-φbest/1 no archive:    v = x + F(xφ − x + x_r1 − x_r3)
        φ = max(round(0.25·NP), 1) best (I:59)
    op3 φbest/1:                          v = F·x_r1 + F(xφ − x_r3)
        φ = max(round(0.5·NP), 2) best (I:66); the code really multiplies
        x_r1 by F (no "+ x_r1" base) — ported as written
  r1 ∈ pop \\ {i}; r2 ∈ pop∪archive \\ {i, r1}; r3 ∈ pop \\ {i, r1, r2}
  (gnR1R2.m, index-value comparison as in the code)
- operator probabilities (I:109-119): improvement ratio
  Δ_i = max(0, f_old − f_new)/|f_old|, S_op = max(0, mean Δ over op's
  individuals); if every S_op ≠ 0 (MATLAB ``if vector`` = all) then
  p_op = max(0.1, min(0.9, S_op/ΣS)) (not renormalised: op3 receives the
  remainder 1 − p1 − p2, possibly 0), else p = 1/3. MATLAB max/min ignore NaN,
  reproduced with np.fmax / np.fmin. → ``allocation="code"`` (default).
  ``allocation="paper"`` instead uses the quality+diversity rule described in
  the paper text (see GUESSES).
- crossover per generation: binomial with prob 0.4, otherwise exponential
  (no wrap-around) (I:75-93)
- CR ~ N(M_CR, 0.1) clipped to [0,1], 0 if M_CR = ⊥; the CR vector is then
  *sorted ascending* and given to the fitness-sorted population (best gets the
  smallest CR) (I:19-23, 38-40). F ~ Cauchy(M_F, 0.1), redrawn while ≤ 0,
  truncated at 1 (I:28-36)
- memory H = 20·D, M_F = M_CR = 0.2 initially (M:55-57); weighted Lehmer means
  with weights ∝ |Δf| (I:133-147); when a generation has no success the current
  slot is reset to 0.5 and the pointer is NOT advanced (I:148-151)
- selection: strict improvement (I:102, 121-123)
- archive: |A| = 2.6·NP (M:49-50, round(2.6·NP) after reductions (M:140));
  improved parents added, duplicates removed, random trimming (updateArchive.m)
- bound handling (han_boun.m): one of three rules drawn per generation
  uniformly: (1) midpoint to parent, (2) the code's "reflection", which
  because of a typo (2·x_L − x on the upper side) sets *every* violating
  coordinate to the lower bound — ported as written, (3) uniform reset using
  a single scalar rand for all violating coordinates
- local search (M:186-210, LS2.m): after every generation, when
  0.85·MaxFES < FEs < MaxFES, with prob p_LS (initially 0.1) run SQP from the
  best-so-far with a budget of min(ceil(0.02·MaxFES), MaxFES − FEs). On
  success the result replaces the worst individual and p_LS = 0.1, otherwise
  p_LS = 0.01
- stopping (M:213): stop once FEs ≥ MaxFES − 4·N_target. The target-hit stop
  (|f − f*| ≤ 1e-8, M:217) is not used — the runner scores the history.

SQP budget accounting
---------------------
fmincon('sqp') is replaced with ``scipy.optimize.minimize(method="SLSQP")``
with finite-difference gradients and the box bounds. Every objective call
SciPy makes (function values and finite-difference probes) goes through the
same ``evaluate`` helper, is appended to the history and counts against
max_evals; when the LS budget (or max_evals) is reached the wrapper raises an
internal exception that aborts SLSQP. The LS result is the best point SLSQP
evaluated (fmincon returns its last iterate; for an SQP with a merit-function
line search that is normally the same point).

GUESSES / deviations
--------------------
- ``allocation="paper"``: reconstructed from the paper's prose (quality rate
  QR_op = mean fitness of op's individuals / Σ, diversity rate DR_op = mean
  distance of op's individuals to the best / Σ, IRV = (1 − QR) + DR, p_op =
  max(0.1, min(0.9, IRV/ΣIRV))). Not verified against the paper's equations;
  the published results come from the code rule, which is the default.
- The CEC 2020 objective includes the bias F*; the repo's functions return the
  error (f* = 0), so the relative improvement Δ/|f_old| is computed on errors.
- SLSQP's default finite-difference step / termination tolerances differ from
  fmincon's (ftol set to 1e-15 here so SLSQP does not stop much earlier than
  fmincon would; guess).
- The budget is checked per evaluation (the code overshoots by up to one
  generation); a partially evaluated generation ends the run.
- RNG: numpy default_rng(seed), not MATLAB's mt19937ar stream.
- The unused CMA-ES branch of the code (Probs = [1 0], M:120) is omitted.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult

_TERMINAL = -1.0


class _BudgetExhausted(Exception):
    pass


def _mround(v: float) -> int:
    """MATLAB round (half away from zero) for v >= 0."""
    return int(np.floor(v + 0.5))


class IMODEOptimizer(BaseOptimizer):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_init_factor: float = 6.0,      # N_init = 6·D²  (P:38)
        n_min: int = 4,                  # (P:41)
        memory_factor: int = 20,         # H = 20·D  (M:55)
        memory_init: float = 0.2,        # (M:56-57)
        arc_rate: float = 2.6,           # (M:49)
        prob_bin: float = 0.4,           # binomial vs exponential (I:75)
        ls_start: float = 0.85,          # LS phase start, fraction of budget (M:186)
        ls_budget_frac: float = 0.02,    # LS budget = ceil(0.02·MaxFES) (LS2.m:7)
        prob_ls_init: float = 0.1,       # (P:44)
        prob_ls_fail: float = 0.01,      # (M:202)
        use_ls: bool = True,
        allocation: str = "code",        # "code" (default) or "paper"
    ):
        super().__init__(benchmark, seed)
        self.n_init = max(n_min, _mround(n_init_factor * self.dim * self.dim))
        self.n_min = n_min
        self.memory_size = int(memory_factor * self.dim)
        self.memory_init = memory_init
        self.arc_rate = arc_rate
        self.prob_bin = prob_bin
        self.ls_start = ls_start
        self.ls_budget_frac = ls_budget_frac
        self.prob_ls_init = prob_ls_init
        self.prob_ls_fail = prob_ls_fail
        self.use_ls = use_ls
        if allocation not in ("code", "paper"):
            raise ValueError("allocation must be 'code' or 'paper'")
        self.allocation = allocation

    # ------------------------------------------------------------------
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        d = self.dim
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        def evaluate(x: np.ndarray) -> float:
            f = float(self.func(x))
            history_x.append(np.array(x, dtype=float, copy=True))
            history_f.append(f)
            return f

        # --- initialisation (M:27-36)
        ps = min(self.n_init, max_evals)
        init_ps = ps
        pop = lo + (hi - lo) * rng.random((ps, d))
        fit = np.array([evaluate(x) for x in pop])
        history_pop.append(pop.copy())
        b = int(np.argmin(fit))
        best_f, best_x = float(fit[b]), pop[b].copy()

        H = self.memory_size
        m_f = np.full(H, self.memory_init)
        m_cr = np.full(H, self.memory_init)
        hist_pos = 0
        probs = np.full(3, 1.0 / 3.0)
        arc_np = self.arc_rate * ps
        archive = np.empty((0, d))
        prob_ls = self.prob_ls_init

        while len(history_f) < max_evals:
            nfe = len(history_f)
            # --- linear population size reduction (M:126-146)
            upd = _mround((self.n_min - init_ps) / max_evals * nfe + init_ps)
            if ps > upd:
                new_ps = max(upd, self.n_min)
                pop, fit = pop[:new_ps], fit[:new_ps]
                ps = new_ps
                arc_np = _mround(self.arc_rate * ps)
                if len(archive) > arc_np:
                    archive = archive[rng.permutation(len(archive))[:arc_np]]

            # --- one IMODE generation (IMODE.m)
            done, n_eval = self._generation(rng, pop, fit, archive, arc_np, probs,
                                            m_f, m_cr, hist_pos, evaluate,
                                            max_evals - len(history_f), lo, hi)
            pop, fit, archive, probs, hist_pos = done
            history_pop.append(pop.copy())
            if fit[0] < best_f:
                best_f, best_x = float(fit[0]), pop[0].copy()
            if n_eval < ps:
                break

            # --- SQP local search phase (M:186-210)
            nfe = len(history_f)
            if (self.use_ls and nfe > self.ls_start * max_evals
                    and nfe < max_evals and rng.random() < prob_ls):
                ls_budget = min(int(np.ceil(self.ls_budget_frac * max_evals)),
                                max_evals - nfe)
                x_ls, f_ls = self._sqp(best_x, ls_budget, evaluate, lo, hi)
                if best_f - f_ls > 0:
                    best_f, best_x = f_ls, x_ls.copy()
                    pop[ps - 1], fit[ps - 1] = x_ls, f_ls
                    order = np.argsort(fit, kind="stable")
                    pop, fit = pop[order], fit[order]
                    prob_ls = self.prob_ls_init
                else:
                    prob_ls = self.prob_ls_fail

            # --- stopping rule of the code (M:213)
            if len(history_f) >= max_evals - 4 * upd:
                break

        return self._make_result(history_x, history_f, history_pop)

    # ------------------------------------------------------------------
    def _generation(self, rng, pop, fit, archive, arc_np, probs, m_f, m_cr,
                    hist_pos, evaluate, remaining, lo, hi):
        ps, d = pop.shape
        H = len(m_f)

        # parameters (I:13-36)
        r = rng.integers(0, H, ps)
        mu_f, mu_cr = m_f[r], m_cr[r]
        cr = rng.normal(mu_cr, 0.1)
        cr[mu_cr == _TERMINAL] = 0.0
        cr = np.clip(cr, 0.0, 1.0)
        F = mu_f + 0.1 * np.tan(np.pi * (rng.random(ps) - 0.5))
        bad = F <= 0
        while bad.any():
            F[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
            bad = F <= 0
        F = np.minimum(F, 1.0)
        order = np.argsort(fit, kind="stable")             # I:38-40
        pop, fit = pop[order], fit[order]
        cr = np.sort(cr)

        # indices (gnR1R2.m)
        pop_all = np.vstack([pop, archive]) if len(archive) else pop
        n_all = len(pop_all)
        r0 = np.arange(ps)
        r1 = rng.integers(0, ps, ps)
        while (m := r1 == r0).any():
            r1[m] = rng.integers(0, ps, m.sum())
        r2 = rng.integers(0, n_all, ps)
        while (m := (r2 == r1) | (r2 == r0)).any():
            r2[m] = rng.integers(0, n_all, m.sum())
        r3 = rng.integers(0, ps, ps)
        while (m := (r3 == r0) | (r3 == r1) | (r3 == r2)).any():
            r3[m] = rng.integers(0, ps, m.sum())

        # operator assignment (I:50-55) — kept and credited after evaluation
        bb = rng.random(ps)
        l2 = probs[0] + probs[1]
        op1 = bb <= probs[0]
        op2 = (bb > probs[0]) & (bb <= l2)
        op3 = (bb > l2) & (bb <= 1.0)

        # mutation (I:59-70)
        Fc = F[:, None]
        vi = np.zeros((ps, d))
        p_np = max(_mround(0.25 * ps), 1)
        phix = pop[rng.integers(0, p_np, ps)]
        vi[op1] = pop[op1] + Fc[op1] * (phix[op1] - pop[op1] + pop[r1[op1]] - pop_all[r2[op1]])
        vi[op2] = pop[op2] + Fc[op2] * (phix[op2] - pop[op2] + pop[r1[op2]] - pop[r3[op2]])
        p_np = max(_mround(0.5 * ps), 2)
        phix = pop[np.minimum(rng.integers(0, p_np, ps), ps - 1)]
        vi[op3] = Fc[op3] * pop[r1[op3]] + Fc[op3] * (phix[op3] - pop[r3[op3]])

        # bound handling (han_boun.m)
        hb = rng.integers(1, 4)
        if hb == 1:
            m = vi < lo
            vi[m] = (pop[m] + lo) / 2.0
            m = vi > hi
            vi[m] = (pop[m] + hi) / 2.0
        elif hb == 2:
            m = vi < lo
            vi[m] = np.minimum(hi, np.maximum(lo, 2 * lo - pop[m]))
            m = vi > hi
            vi[m] = np.maximum(lo, np.minimum(hi, 2 * lo - pop[m]))  # code's typo
        else:
            m = vi < lo
            vi[m] = lo + rng.random() * (hi - lo)
            m = vi > hi
            vi[m] = lo + rng.random() * (hi - lo)

        # crossover (I:75-93)
        if rng.random() < self.prob_bin:
            mask = rng.random((ps, d)) > cr[:, None]       # True -> from parent
            mask[r0, rng.integers(0, d, ps)] = False
            ui = np.where(mask, pop, vi)
        else:
            ui = pop.copy()
            start = rng.integers(0, d, ps)
            for i in range(ps):
                l = start[i]
                while rng.random() < cr[i] and l < d - 1:
                    l += 1
                ui[i, start[i]:l + 1] = vi[i, start[i]:l + 1]

        # evaluation (budget checked per evaluation)
        n_eval = min(ps, remaining)
        fit_new = np.full(ps, np.inf)
        for i in range(n_eval):
            fit_new[i] = evaluate(ui[i])

        diff = np.abs(fit - fit_new)
        imp = fit_new < fit
        good_cr, good_f = cr[imp], F[imp]

        # archive (updateArchive.m)
        if arc_np > 0 and imp.any():
            cand = np.vstack([archive, pop[imp]]) if len(archive) else pop[imp].copy()
            cand = np.unique(cand, axis=0)
            if len(cand) > arc_np:
                cand = cand[rng.permutation(len(cand))[:int(np.ceil(arc_np))]]
            archive = cand

        # operator probabilities (I:109-119)
        new_pop = pop.copy()
        new_fit = fit.copy()
        new_fit[imp] = fit_new[imp]
        new_pop[imp] = ui[imp]
        masks = (op1, op2, op3)
        if self.allocation == "code":
            with np.errstate(divide="ignore", invalid="ignore"):
                diff2 = np.fmax(0.0, fit - fit_new) / np.abs(fit)
            if n_eval < ps:
                diff2[n_eval:] = 0.0
            count = np.array([np.fmax(0.0, np.mean(diff2[mk])) if mk.any() else 0.0
                              for mk in masks])
            # MATLAB mean([]) = NaN, max(0, NaN) = 0 -> 0 for empty operators
            if np.all(count != 0):
                with np.errstate(invalid="ignore"):
                    probs = np.fmax(0.1, np.fmin(0.9, count / np.sum(count)))
            else:
                probs = np.full(3, 1.0 / 3.0)
        else:
            probs = self._paper_probs(new_pop, new_fit, masks)

        # memory update (I:127-152)
        if good_cr.size > 0:
            w = diff[imp] / np.sum(diff[imp])
            m_f[hist_pos] = np.sum(w * good_f ** 2) / np.sum(w * good_f)
            if good_cr.max() == 0 or m_cr[hist_pos] == _TERMINAL:
                m_cr[hist_pos] = _TERMINAL
            else:
                m_cr[hist_pos] = np.sum(w * good_cr ** 2) / np.sum(w * good_cr)
            hist_pos = (hist_pos + 1) % H
        else:
            m_cr[hist_pos] = 0.5
            m_f[hist_pos] = 0.5

        order = np.argsort(new_fit, kind="stable")          # I:155-157
        return (new_pop[order], new_fit[order], archive, probs, hist_pos), n_eval

    @staticmethod
    def _paper_probs(pop, fit, masks):
        """Quality + diversity rule as described in the paper text (GUESS)."""
        best = pop[int(np.argmin(fit))]
        qual = np.zeros(3)
        div = np.zeros(3)
        for k, mk in enumerate(masks):
            if mk.any():
                qual[k] = np.mean(fit[mk])
                div[k] = np.mean(np.linalg.norm(pop[mk] - best, axis=1))
        if not all(mk.any() for mk in masks) or qual.sum() <= 0 or div.sum() <= 0:
            return np.full(3, 1.0 / 3.0)
        irv = (1.0 - qual / qual.sum()) + div / div.sum()
        return np.fmax(0.1, np.fmin(0.9, irv / irv.sum()))

    # ------------------------------------------------------------------
    def _sqp(self, x0, budget, evaluate, lo, hi):
        """SQP from x0 with at most ``budget`` objective calls (LS2.m)."""
        best = {"x": np.array(x0, dtype=float), "f": np.inf, "n": 0}

        def fun(x):
            if best["n"] >= budget:
                raise _BudgetExhausted
            xc = np.clip(x, lo, hi)
            f = evaluate(xc)
            best["n"] += 1
            if f < best["f"]:
                best["f"], best["x"] = f, xc.copy()
            return f

        if budget <= 0:
            return best["x"], np.inf
        try:
            minimize(fun, np.array(x0, dtype=float), method="SLSQP",
                     bounds=[(lo, hi)] * self.dim,
                     options={"maxiter": 10 ** 6, "ftol": 1e-15})
        except _BudgetExhausted:
            pass
        return best["x"], best["f"]

