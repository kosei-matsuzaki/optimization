"""The memoryless multistart null, as a *run* rather than as an estimate.

Everything the project says about the GECCO'2024 suite currently rests on one
number: 0.5529, the 16-problem mean MPR of the "restart lander" null.  That
number was never produced by a run.  It is a multinomial calculation
(``analysis/mmo2024/e103/analyze.py``) over 500 *offline* descents per problem:
``n' = budget / mean descent cost`` restarts are assumed, each drawing
independently from the observed landing distribution, and the coverage is
``sum_j (1 - (1-p_j)^n') / K``.  Neither the harness nor the reported-set cap
nor the scorer is on that path.

This class puts the same null on the ordinary path.  One run is a chain of
descents, laid end to end inside the suite's own budget:

  draw x0 uniformly from the box
    -> isotropic CMA-ES descent (``CMA_on = 0``) from x0 with
       ``sigma0 = sigma_ratio * span``, stopped by CMA's own criteria or by
       ``descent_budget`` evaluations, whichever comes first
    -> the descent's best point joins ``final_solutions``
  repeat until the run budget is spent.

The descent is byte-for-byte the one ``scripts/hunt_coverage.py:_null_descent``
draws offline (same ``tolfun = tolfunhist = tolx = 0``, same isotropic
covariance, same per-descent cap), so the only differences against the estimate
are the ones the estimate abstracts away: the restart count is whatever the
budget actually buys instead of ``budget / mean cost``, the reported set goes
through ``max(100, 2K)`` best-by-f capping in ``core.runner._niching_counts``,
and the landings of one run are the landings of one run rather than an i.i.d.
draw from a 500-descent empirical distribution.

Defaults are entry 103's, because 0.5529 is entry 103's: ``sigma_ratio = 0.1``
and ``descent_budget = 12500`` (the suite budget / 40).  This is a *null*, not a
proposal -- it has no memory between restarts, no repulsion and no selection
over draws, which is the whole point of comparing a niching method against it.

Set ``RESTART_LANDER_DUMP`` to a directory and each run writes one row per
descent (landing optimum, best f, evaluations, CMA's stop reason) there.  The
stop reasons matter: with all three tolerances at 0 it is not obvious what ends
a descent, and "the descent stops earlier than the estimate charges it for" is
one of the ways the estimate could be wrong.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

import numpy as np

from .base import BaseOptimizer, OptimizeResult


class RestartLanderOptimizer(BaseOptimizer):
    """Uniform restart + isotropic local descent, chained to fill the budget."""

    def __init__(self, benchmark, seed: int = 42, sigma_ratio: float = 0.1,
                 descent_budget: int = 12500, iso: bool = True):
        super().__init__(benchmark, seed)
        self.sigma_ratio = float(sigma_ratio)
        self.descent_budget = int(descent_budget)
        self.iso = bool(iso)
        # one (landing, best_f, evals, stop-reason) record per descent
        self.descents: list[dict] = []

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        import cma
        lo, hi = self.bounds
        sigma0 = self.sigma_ratio * (hi - lo)
        # `rng_override` exists for the identity check only: it lets a test
        # feed this run the exact draws `_null_descent` used offline, so the
        # two descents can be compared point for point.
        rng = getattr(self, "rng_override", None) or np.random.default_rng(self.seed)
        hx: list[np.ndarray] = []
        hf: list[float] = []
        reported: list[np.ndarray] = []
        used = 0
        k = 0
        while used < max_evals:
            x0 = rng.uniform(lo, hi, size=self.dim)
            cap = min(self.descent_budget, max_evals - used)
            o = {"bounds": [lo, hi], "maxfevals": cap, "seed": self.seed + k + 1,
                 "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolx": 0}
            if self.iso:
                o["CMA_on"] = 0                  # step size only, no rotation
            es = cma.CMAEvolutionStrategy(list(x0), sigma0, o)
            best_f, best_x, spent = float("inf"), x0, 0
            while not es.stop():
                xs = es.ask()
                if used + len(xs) > max_evals:   # never overspend the budget
                    break
                fs = [float(self.func(np.asarray(x))) for x in xs]
                used += len(xs)
                spent += len(xs)
                es.tell(xs, fs)
                hx.extend(np.asarray(x, dtype=float) for x in xs)
                hf.extend(fs)
                i = int(np.argmin(fs))
                if fs[i] < best_f:
                    best_f, best_x = fs[i], np.asarray(xs[i], dtype=float)
            if spent == 0:                       # budget exhausted mid-descent
                break
            reported.append(best_x.copy())
            self.descents.append({
                "descent": k, "evals": spent, "best_f": best_f,
                "stop": "|".join(sorted(es.stop())) or "budget",
                "x": best_x.copy(),
            })
            k += 1
        self._dump()
        return self._make_result(hx, hf, solutions=reported)

    def _dump(self) -> None:
        d = os.environ.get("RESTART_LANDER_DUMP")
        if not d or not self.descents:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        opts = np.asarray(self.benchmark.optima_pos, dtype=float)
        path = out / f"{self.benchmark.name}_seed{self.seed}.csv"
        # `land_opt`/`dist` need the true optima, so they are oracle columns --
        # diagnostics only.  The `x*` columns are what the algorithm itself
        # holds, so a reporting rule computed from them is legal under the
        # competition's parameter rules (entry 112 section 6).
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["descent", "evals", "best_f", "land_opt", "dist", "stop"]
                       + [f"x{i}" for i in range(self.dim)])
            for r in self.descents:
                dd = np.linalg.norm(opts - r["x"], axis=1)
                j = int(np.argmin(dd))
                w.writerow([r["descent"], r["evals"], f"{r['best_f']:.12g}",
                            j, f"{dd[j]:.6g}", r["stop"]]
                           + [f"{v:.12g}" for v in r["x"]])
