"""Restart-CMA-ES baselines: IPOP (Auger & Hansen 2005) and BIPOP (Hansen 2009).

Both wrap pycma under the BaseOptimizer interface so they slot directly into
the existing runner without changes.
"""
from __future__ import annotations

import math

import numpy as np
import cma

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _RestartCMAESBase(BaseOptimizer):
    """Shared restart-CMA-ES driver. Subclasses set the regime."""

    bipop: bool = False
    incpopsize: int = 2
    repelling: bool = False

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        sigma0: float = 1.0,
        repel_coverage: float = 0.2,   # share of the box the taboo balls may block
        repel_gamma: float = 0.9,      # radius shrink per rejection of the same point
        repel_max_resample: int = 10,  # give up and accept after this many redraws
    ):
        super().__init__(benchmark, seed)
        self.sigma0 = sigma0
        self.repel_coverage = repel_coverage
        self.repel_gamma = repel_gamma
        self.repel_max_resample = repel_max_resample

    def _taboo_radius(self, n_taboo: int) -> float:
        """Radius of the d-ball that gives each taboo point an equal share of the
        blocked volume: total blocked volume = repel_coverage × box volume."""
        lo, hi = self.bounds
        volume = (hi - lo) ** self.dim * self.repel_coverage / max(n_taboo, 1)
        return (volume * math.gamma(self.dim / 2 + 1)) ** (1 / self.dim) / math.sqrt(math.pi)

    def _repel(self, solutions: list, taboo: list[np.ndarray],
               n_rej: list[int], es) -> list:
        """Redraw candidates that land in a taboo ball; shrink that ball each
        time it rejects, so a stubbornly-blocked region eventually opens up."""
        if not taboo:
            return solutions
        radius = self._taboo_radius(len(taboo))
        out = []
        for s in solutions:
            x = np.asarray(s, dtype=float)
            for _ in range(self.repel_max_resample):
                hit = next((j for j, t in enumerate(taboo)
                            if float(np.linalg.norm(x - t))
                            < radius * self.repel_gamma ** n_rej[j]), None)
                if hit is None:
                    break
                n_rej[hit] += 1
                x = np.asarray(es.ask(1)[0], dtype=float)
            out.append(x)
        return out

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        # default popsize for CMA-ES; doubles each IPOP restart
        lambda0 = 4 + int(3 * np.log(self.dim))

        # Track # FEs consumed in "large" vs "small" regime (BIPOP)
        fe_large = 0
        fe_small = 0
        restart_idx = 0
        restart_bests: list[np.ndarray] = []
        taboo_rejections: list[int] = []
        # Initial sigma & x0
        x0 = rng.uniform(lo, hi, self.dim)
        sigma = self.sigma0

        while len(history_f) < max_evals:
            remaining = max_evals - len(history_f)

            # Choose popsize for this restart
            if restart_idx == 0:
                popsize = lambda0
                regime = "large"
            elif self.bipop:
                # BIPOP: alternate to balance large vs small budgets
                if fe_small < fe_large:
                    # Small regime: λ uniformly in [λ0, λ_large/2]
                    lambda_large = lambda0 * (self.incpopsize ** restart_idx)
                    upper = max(lambda0, lambda_large // 2)
                    popsize = int(rng.integers(lambda0, upper + 1))
                    sigma = self.sigma0 * 10 ** (-2 * float(rng.random()))
                    regime = "small"
                else:
                    popsize = lambda0 * (self.incpopsize ** restart_idx)
                    sigma = self.sigma0
                    regime = "large"
            else:
                # IPOP: monotonically increasing
                popsize = lambda0 * (self.incpopsize ** restart_idx)
                sigma = self.sigma0
                regime = "large"

            opts = cma.CMAOptions()
            opts["seed"] = int(self.seed) + 1000 * restart_idx
            opts["bounds"] = [[lo] * self.dim, [hi] * self.dim]
            opts["maxfevals"] = remaining
            opts["popsize"] = int(popsize)
            opts["verbose"] = -9

            # x0 for this restart: random uniform (standard IPOP/BIPOP convention)
            x0_this = rng.uniform(lo, hi, self.dim)
            es = cma.CMAEvolutionStrategy(x0_this, sigma, opts)

            fe_before = len(history_f)
            while len(history_f) < max_evals and not es.stop():
                solutions = es.ask()
                if self.repelling:
                    solutions = self._repel(solutions, restart_bests, taboo_rejections, es)
                fitnesses = [float(self.func(np.asarray(s))) for s in solutions]
                es.tell(solutions, fitnesses)
                history_pop.append(np.array(solutions))
                for s, f in zip(solutions, fitnesses):
                    history_x.append(np.asarray(s))
                    history_f.append(f)
            fe_used = len(history_f) - fe_before
            if fe_used > 0:
                local = int(np.argmin(history_f[fe_before:])) + fe_before
                restart_bests.append(history_x[local].copy())
                taboo_rejections.append(0)
            if regime == "large":
                fe_large += fe_used
            else:
                fe_small += fe_used
            restart_idx += 1

        # Each restart searches a fresh basin, so its best point is a distinct
        # candidate solution; the last population is reported alongside them.
        solutions = restart_bests + (list(history_pop[-1]) if history_pop else [])
        return self._make_result(history_x, history_f, history_pop,
                                 solutions=solutions or None)


class IPOPCMAESOptimizer(_RestartCMAESBase):
    """IPOP-CMA-ES (Auger & Hansen, 2005). λ doubles each restart."""
    bipop = False


class BIPOPCMAESOptimizer(_RestartCMAESBase):
    """BIPOP-CMA-ES (Hansen, 2009). Alternates between large and small λ regimes."""
    bipop = True


class RepellingCMAESOptimizer(_RestartCMAESBase):
    """IPOP-CMA-ES that refuses to fall back into a basin it already drilled.

    Approximates de Nobel, Vermetten, Kononova, Shir & Bäck (PPSN 2024),
    "Avoiding Redundant Restarts in Multimodal Global Optimization": the best
    point of every finished restart becomes a taboo point, candidates drawn
    inside a taboo ball are redrawn, and a ball that keeps rejecting shrinks by
    ``repel_gamma`` per rejection so the search can still get back in when it
    genuinely needs to. The radius follows their volume argument — the taboo set
    may block ``repel_coverage`` of the box, split evenly over the taboo points,
    so radii shrink automatically as restarts pile up.

    Deviation from the paper: rejection is tested in plain Euclidean distance,
    while they test Mahalanobis distance in the current CMA metric divided by σ.
    This is a reimplementation of the idea, not their code. ``repel_coverage``
    is our choice too (the paper's value is not in the text we could read), and
    it is the parameter that decides whether repelling helps or strangles the
    search — the paper reports aggressive repelling hurting structured
    landscapes — so it needs a sensitivity check before any claim rests on it.

    This is the baseline MC-ESO's informed restart has to be measured against —
    basin memory plus repulsion is exactly what spillover does, so without this
    row in the table there is no way to say what MC-ESO adds on top of a
    published repelling restart.
    """
    bipop = False
    repelling = True
