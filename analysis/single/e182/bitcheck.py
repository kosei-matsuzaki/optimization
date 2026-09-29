"""Bit-identity checks for the cc_mu_frac arm (その182).

(1) refactor check: the refactored _update_cc_cov must leave the DEFAULT
    MC-ESO bit-identical at dim 10 (where the learned-C gate is fully open).
    Run with --ref pointing at a pre-edit worktree to produce the reference.
(2) arm check: ccmu50 must be bit-identical to MC-ESO at dim 2, because
    _cc_dim_gate() is exactly 0 there and the learned C is never sampled.
"""
import json, sys
import numpy as np
from core.benchmarks import BENCHMARKS_BY_NAME, BENCHMARKS_10D_BY_NAME
from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M

def best_f(bench, kwargs, n_runs, max_evals):
    out = []
    for i in range(n_runs):
        opt = M(bench, seed=i * 100, **kwargs)
        r = opt.optimize(max_evals=max_evals)
        out.append(repr(float(r.best_f)))
    return out

if __name__ == "__main__":
    mode = sys.argv[1]
    res = {}
    if mode in ("d10", "all"):
        for fn in ("F07-StepEllipsoidal", "F12-BentCigar"):
            b = BENCHMARKS_10D_BY_NAME[fn]
            res[f"d10/default/{fn}"] = best_f(b, {}, 3, 25000)
    if mode in ("d2", "all"):
        for fn in ("F17-SchafferF7", "F01-Sphere"):
            b = BENCHMARKS_BY_NAME[fn]
            res[f"d2/default/{fn}"] = best_f(b, {}, 3, 5000)
            res[f"d2/ccmu50/{fn}"] = best_f(b, {"cc_mu_frac": 0.50}, 3, 5000)
    print(json.dumps(res, indent=0, sort_keys=True))
