"""Can the five stalled basins of N18-CF3-10D be descended at all?

`descent_probe.py` measures the shape of the descent MC-ESO actually gets. This
asks the counterfactual the shape cannot answer: take the best point a stalled
hunt reached and hand it to a local optimiser with a fresh budget. Three
outcomes, each with a different consequence:

  * an anisotropic local search (CMA-ES, covariance on) descends while an
    isotropic one (same code, `CMA_on=0`, so only the step size adapts) does
    not  -> the stall is the isotropic step MC-ESO drills with, on a rotated /
    ill-conditioned CF3 component. Entry 75 stands and the fix is named.
  * neither descends -> the point is at a *local* optimum, so the 6/6 coverage
    at eps = 1e-1 (the scorer only asks for six points rho apart with f <= 0.1)
    is not six global basins, and status.md's "coverage 6/6, depth 1/6" reading
    has to be rewritten.
  * both descend -> the basin was descendable and MC-ESO stopped for reasons of
    its own, which is shape (ii) and contradicts entry 75.

The run is deterministic in the same way the probe is (same seed -> identical
trajectory), so the start points are exactly the ones the measured run reached.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from descent_probe import _DescentProbe, _attribute                 # noqa: E402


def _cma_run(b, x0: np.ndarray, sigma0: float, budget: int, seed: int,
             iso: bool) -> tuple[float, int]:
    import cma
    lo, hi = b.bounds
    opts = {"bounds": [lo, hi], "maxfevals": budget, "seed": seed + 1,
            "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolx": 0}
    if iso:
        opts["CMA_on"] = 0          # step-size adaptation only, no rotation
    es = cma.CMAEvolutionStrategy(list(np.asarray(x0, dtype=float)), sigma0, opts)
    best, used = float("inf"), 0
    while not es.stop() and used < budget:
        xs = es.ask()
        fs = [float(b.func(np.asarray(x))) for x in xs]
        used += len(xs)
        es.tell(xs, fs)
        best = min(best, min(fs))
    return best, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="N18-CF3-10D")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20000, help="per local run")
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    ev_budget = a.evals or int(b.suite_max_evals)
    opt = np.asarray(b.optima_pos, dtype=float)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["seed", "gopt", "dist_start", "f_start", "f_cma_full",
                "f_cma_iso", "evals_full", "evals_iso"])

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        t0 = time.time()
        o = _DescentProbe(b, seed=s * 100)
        o.optimize(ev_budget)
        if o._cur is not None:
            o._cur["bb_end"] = min(o._cur["bb"]) if o._cur["bb"] else float("nan")
            o.hunts.append(o._cur)

        #  best point per true global optimum, over every hunt of this run
        best: dict[int, tuple[float, np.ndarray, float]] = {}
        for c in o.hunts:
            if c["bx"] is None or not c["bb"]:
                continue
            j, d = _attribute(c["bx"], opt)
            fv = float(min(c["bb"]))
            if j not in best or fv < best[j][0]:
                best[j] = (fv, c["bx"], d)
        print(f"seed={s} probe={time.time() - t0:.0f}s "
              f"optima_touched={sorted(best)}", flush=True)

        for j in sorted(best):
            fv, x0, d = best[j]
            ff, ef = _cma_run(b, x0, a.sigma0, a.budget, s, iso=False)
            fi, ei = _cma_run(b, x0, a.sigma0, a.budget, s, iso=True)
            w.writerow([s, j, f"{d:.4g}", f"{fv:.6g}", f"{ff:.6g}",
                        f"{fi:.6g}", ef, ei])
            f.flush()
            print(f"  seed={s} gopt={j} d={d:.3g} f0={fv:.4g} "
                  f"full={ff:.4g} iso={fi:.4g}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
