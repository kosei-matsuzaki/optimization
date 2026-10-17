"""Control optimizers — not methods under study.

They exist to answer one question about MC-ESO: does its own machinery do the
work, or would a naive combination of known parts match it?
"""
from __future__ import annotations

import numpy as np

from .restart_cmaes import IPOPCMAESOptimizer
from .base import OptimizeResult


class SepScanIPOPOptimizer(IPOPCMAESOptimizer):
    """IPOP-CMA-ES behind the same separability test and coordinate scan as
    MC-ESO's reassort_scan (2026-10-17 control).

    From a uniform point, a few coordinate pairs are tested for additivity
    (|Δ_ij − (Δ_i + Δ_j)| ≤ tol·(|Δ_i| + |Δ_j|)); if every pair passes, each
    coordinate is scanned on a `grid`-point grid and refined by golden-section
    search, for up to two passes (as MC-ESO: at most 2·D·(grid + 61)
    evaluations). IPOP-CMA-ES then spends the rest of the budget, its first
    restart starting from the scan's best point. MC-ESO runs the test only
    after a stall; here it runs first, which only helps the control.
    """

    def __init__(self, benchmark, seed: int = 42, sigma0: float = 1.0,
                 grid: int = 200, refine: int = 61, tol: float = 0.02, pairs: int = 3):
        super().__init__(benchmark, seed=seed, sigma0=sigma0)
        self.grid, self.refine, self.tol, self.pairs = grid, refine, tol, pairs

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed + 7)
        lo, hi = self.bounds
        hx: list[np.ndarray] = []
        hf: list[float] = []

        def ev(x):
            f = float(self.func(x))
            hx.append(x.copy())
            hf.append(f)
            return f

        x = rng.uniform(lo, hi, self.dim)
        fx = ev(x)
        separable = self.dim >= 2
        for _ in range(self.pairs if separable else 0):
            i, j = rng.choice(self.dim, 2, replace=False)
            vi, vj = rng.uniform(lo, hi, 2)
            xi = x.copy(); xi[i] = vi
            xj = x.copy(); xj[j] = vj
            xij = xi.copy(); xij[j] = vj
            di, dj, dij = ev(xi) - fx, ev(xj) - fx, ev(xij) - fx
            if abs(dij - (di + dj)) > self.tol * (abs(di) + abs(dj)):
                separable = False
                break
        cap = 2 * self.dim * (self.grid + self.refine)
        if separable and len(hf) + cap < max_evals:
            g = np.linspace(lo, hi, self.grid)
            phi = (np.sqrt(5) - 1) / 2
            for _ in range(2):
                moved = False
                for d in range(self.dim):
                    best_v, best_f = x[d], fx
                    for v in g:
                        y = x.copy(); y[d] = v
                        fy = ev(y)
                        if fy < best_f:
                            best_v, best_f = v, fy
                    h = (hi - lo) / (self.grid - 1)
                    a, b = max(lo, best_v - h), min(hi, best_v + h)
                    for _ in range(self.refine // 2):
                        c, e = b - phi * (b - a), a + phi * (b - a)
                        yc = x.copy(); yc[d] = c
                        ye = x.copy(); ye[d] = e
                        fc, fe = ev(yc), ev(ye)
                        for v, fv in ((c, fc), (e, fe)):
                            if fv < best_f:
                                best_v, best_f = v, fv
                        if fc < fe:
                            b = e
                        else:
                            a = c
                    if best_f < fx:
                        x[d], fx, moved = best_v, best_f, True
                if not moved:
                    break
        left = max_evals - len(hf)
        if left <= 0:
            return self._make_result(hx, hf, [np.array(hx[-1:])])
        self.x0_first = x.copy()
        try:
            res = IPOPCMAESOptimizer.optimize(self, left)
        finally:
            self.x0_first = None
        return self._make_result(hx + list(res.history_x), hf + list(res.history_f),
                                 [np.array(hx[-1:])] + list(res.history_pop))
