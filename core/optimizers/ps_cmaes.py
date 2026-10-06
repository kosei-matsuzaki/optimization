"""PS-CMA-ES — a swarm of CMA-ES instances steered toward the global best.

    C. L. Müller, B. Baumgartner, I. F. Sbalzarini, "Particle swarm CMA
    evolution strategy for the optimization of multi-funnel landscapes",
    IEEE CEC 2009, pp. 2685–2692.

Self-contained numpy CMA-ES (Hansen's standard rank-one + rank-μ + CSA) so the
swarm step can rotate each instance's covariance matrix directly.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


def _givens_to_axis(v: np.ndarray) -> np.ndarray:
    """Orthogonal R (product of Givens rotations, Algorithm 1 loop order)
    with R v = ||v|| e_1."""
    n = len(v)
    R = np.eye(n)
    p = v.astype(float).copy()
    for i in range(n - 2, -1, -1):          # i = n-1 … 1 (1-based)
        for j in range(n - 1, i, -1):       # j = n … i+1 (1-based)
            a, b = p[i], p[j]
            r = np.hypot(a, b)
            if r == 0.0:
                continue
            c, s = a / r, b / r
            # rotate rows i, j of p and of R
            pi, pj = c * p[i] + s * p[j], -s * p[i] + c * p[j]
            p[i], p[j] = pi, pj
            Ri, Rj = c * R[i] + s * R[j], -s * R[i] + c * R[j]
            R[i], R[j] = Ri, Rj
    return R


def rotation_onto(b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Algorithm 1 of Müller et al. (2009): R = R_p^T R_b, so R b ∝ p."""
    return _givens_to_axis(p).T @ _givens_to_axis(b)


class _CMA:
    """Minimal (μ/μ_w, λ)-CMA-ES state (Hansen 2016 tutorial defaults)."""

    def __init__(self, mean: np.ndarray, sigma: float, lam: int):
        n = len(mean)
        self.n = n
        self.lam = lam
        self.mu = lam // 2
        w = np.log((lam - 1) / 2 + 1) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.reset(mean, sigma)

    def reset(self, mean: np.ndarray, sigma: float) -> None:
        n = self.n
        self.m = mean.astype(float).copy()
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.Dv = np.ones(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.gen = 0

    def eig(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, B = np.linalg.eigh(self.C)
        d2 = np.maximum(d2, 1e-300)
        self.Dv = np.sqrt(d2)
        self.B = B

    def ask(self, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * (z * self.Dv) @ self.B.T

    def tell(self, X: np.ndarray, f: np.ndarray) -> None:
        n = self.n
        idx = np.argsort(f, kind="stable")[: self.mu]
        Y = (X[idx] - self.m) / self.sigma
        yw = self.w @ Y
        self.m = self.m + self.sigma * yw
        self.gen += 1
        invsqrtC_yw = self.B @ ((self.B.T @ yw) / self.Dv)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_yw
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                / self.chiN) < (1.4 + 2 / (n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        dh = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp(min(1.0, (self.cs / self.damps)
                                 * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.eig()


class PSCMAESOptimizer(BaseOptimizer):
    """PS-CMA-ES (Müller, Baumgartner & Sbalzarini, CEC 2009).

    S CMA-ES instances run in lock-step generations. Every ``comm_interval``
    generations (I_c) the global best g is broadcast and every instance

    1. mixes its covariance with a copy rotated so that its principal
       eigenvector points along p_g = g − m (Eq. 7–8, Algorithm 1, Givens):
       C ← c_p·C + (1 − c_p)·R C Rᵀ;
    2. biases its mean (Algorithm 2): the instance owning g and instances with
       σ ≥ ||p_g|| get no bias; if σ/||p_g|| ≤ t_c the bias is b·p_g (converged
       far away → jump); otherwise (σ/||p_g||)·p_g.

    Paper standard setting (Sec. IV-B, 10-D CEC 2005 grid search): S = 15,
    c_p = 0.7, I_c = 200, σ0 = 0.2·range, t_c = 0.1, b = 0.5; per-instance
    CMA-ES defaults from Hansen (λ = 4 + ⌊3 ln n⌋).

    Notes / guesses:
    * Algorithm 2's convergence test is garbled in the PDF; read here as the
      dimensionless σ/||p_g|| ≤ t_c.
    * With I_c = 200 a swarm generation costs S·λ evaluations (150 at D = 10),
      so budgets below ~S·λ·I_c (30 000 at D = 10, 18 000 at D = 2) never
      reach a communication step and the method degenerates to S independent
      CMA-ES runs — exactly as the paper's I_c → ∞ limit. Pass a smaller
      ``comm_interval`` for short budgets (documented deviation if used).
    * The paper runs no restarts. To keep instances numerically sane, an
      instance whose step σ·max√eig(C) falls below ``restart_tol``·range or whose
      condition number exceeds 1e14 is re-initialised uniformly (our addition).
    * Box constraints: samples are clipped to the box and the clipped point
      is used in the update (repair; the paper does not specify).
    * Starting means uniform in the box.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        swarm_size: int = 15,          # paper S
        cp: float = 0.7,               # paper c_p
        comm_interval: int = 200,      # paper I_c (generations)
        sigma0_frac: float = 0.2,      # paper σ0 = 0.2 · range
        tc: float = 0.1,               # paper t_c
        bias_b: float = 0.5,           # paper b
        popsize: int | None = None,    # None → 4 + ⌊3 ln n⌋ (Hansen default)
        restart_tol: float = 1e-12,    # our addition (see docstring)
    ):
        super().__init__(benchmark, seed)
        self.swarm_size = int(swarm_size)
        self.cp = cp
        self.comm_interval = int(comm_interval)
        self.sigma0_frac = sigma0_frac
        self.tc = tc
        self.bias_b = bias_b
        self.popsize = popsize or 4 + int(3 * np.log(benchmark.dim))
        self.restart_tol = restart_tol

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = map(float, self.bounds)
        span = hi - lo
        sigma0 = self.sigma0_frac * span

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        swarm = [_CMA(rng.uniform(lo, hi, self.dim), sigma0, self.popsize)
                 for _ in range(self.swarm_size)]
        g_x: np.ndarray | None = None
        g_f = np.inf
        g_owner = -1
        gen = 0

        while len(history_f) < max_evals:
            gen_samples = []
            for k, es in enumerate(swarm):
                if len(history_f) >= max_evals:
                    break
                X = np.clip(es.ask(rng), lo, hi)
                n_ok = min(len(X), max_evals - len(history_f))
                f = np.empty(len(X))
                for j in range(n_ok):
                    f[j] = float(self.func(X[j]))
                    history_x.append(X[j].copy())
                    history_f.append(f[j])
                    if f[j] < g_f:
                        g_f, g_x, g_owner = f[j], X[j].copy(), k
                gen_samples.append(X[:n_ok])
                if n_ok < len(X):
                    break  # budget exhausted mid-generation
                es.tell(X, f)
                step = es.sigma * es.Dv.max()
                if (step < self.restart_tol * span
                        or es.Dv.max() ** 2 > 1e14 * es.Dv.min() ** 2
                        or not np.all(np.isfinite(es.C))):
                    es.reset(rng.uniform(lo, hi, self.dim), sigma0)
            if gen_samples:
                history_pop.append(np.vstack(gen_samples))
            gen += 1

            if gen % self.comm_interval == 0 and g_x is not None:
                for k, es in enumerate(swarm):
                    pg = g_x - es.m
                    npg = float(np.linalg.norm(pg))
                    if npg == 0.0:
                        continue
                    # (1) covariance rotation toward the global best
                    b_main = es.B[:, int(np.argmax(es.Dv))]
                    if b_main @ pg < 0:
                        b_main = -b_main
                    R = rotation_onto(b_main, pg)
                    es.C = self.cp * es.C + (1 - self.cp) * (R @ es.C @ R.T)
                    es.eig()
                    # (2) mean bias (Algorithm 2)
                    if k == g_owner or es.sigma >= npg:
                        continue
                    ratio = es.sigma / npg
                    es.m = es.m + (self.bias_b if ratio <= self.tc else ratio) * pg

        solutions = [es.m.copy() for es in swarm]
        if history_pop:
            solutions += list(history_pop[-1])
        return self._make_result(history_x, history_f, history_pop, solutions=solutions)
