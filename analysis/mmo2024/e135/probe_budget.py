#!/usr/bin/env python3
"""e135 diagnostic: is the shortfall in the harness's accounting/reporting, or
in the search itself?

Refutation (a) fired on all three functions (analyze.py). Before attributing it
to a harness bug, three things are checked on one run, at zero extra design
freedom:

1. **Budget accounting** — how many real objective calls the wrapper actually
   made against the 400,000 it was given. `core/optimizers/nmmso.py:59-60`
   stops evaluating once `len(history_f) >= max_evals`, and pynmmso ends the
   run on its *own* counter, so the two can disagree.
2. **Reporting loss** — modes returned by NMMSO, modes surviving the
   `np.isfinite` filter, and the size after the harness's `max(100, 2K)` cap.
   If the cap is binding, reported-set truncation is a candidate mechanism.
3. **Search reach** — for each global optimum, the smallest `f` the *whole*
   evaluation history ever attained within that optimum's rho-ball. This is
   the supremum over every reporting rule: if the history never gets near an
   optimum, no reporting rule could have reported it and the loss is search,
   not bookkeeping.

Recording only. No optimizer parameter is changed.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import niching_by_name                     # noqa: E402
from core.optimizers import NMMSOOptimizer                      # noqa: E402

FUNCS = sys.argv[1:] or ["N19-CF4-10D"]

for name in FUNCS:
    b = niching_by_name(name)
    budget = int(b.suite_max_evals)
    opt = NMMSOOptimizer(b, seed=0)
    t0 = time.time()
    res = opt.optimize(budget)
    dt = time.time() - t0

    hx = np.asarray(res.history_x, dtype=float)
    hf = np.asarray(res.history_f, dtype=float)
    sols = np.asarray(res.final_solutions, dtype=float)
    opts = np.asarray(b.optima_pos, dtype=float)
    rho = float(b.niche_rho)
    cap = max(100, 2 * b.n_global_optima)

    print(f"\n=== {name}  D={b.dim}  K={b.n_global_optima}  rho={rho}  "
          f"budget={budget}  ({dt:.0f}s) ===")
    print(f"1. budget accounting : real objective calls = {len(hf)} "
          f"/ {budget} given   ({len(hf) / budget:.4f} of budget)")
    print(f"2. reporting         : modes kept by wrapper = {len(sols)}, "
          f"harness cap = {cap}, cap binding = {len(sols) > cap}")

    # 3. per-optimum reach, from the whole history (supremum over reporting rules)
    d = np.linalg.norm(hx[:, None, :] - opts[None, :, :], axis=2)
    print("3. search reach per optimum (whole history, uncapped):")
    print(f"   {'opt':>4}{'min dist':>12}{'best f in rho-ball':>22}"
          f"{'pts in rho-ball':>18}")
    reached = {a: 0 for a in (1e-1, 1e-3, 1e-5)}
    for k in range(len(opts)):
        inside = d[:, k] <= rho
        best_in = hf[inside].min() if inside.any() else float("inf")
        print(f"   {k:>4}{d[:, k].min():>12.4f}{best_in:>22.6g}"
              f"{int(inside.sum()):>18}")
        for a in reached:
            if best_in <= a:
                reached[a] += 1
    K = b.n_global_optima
    print("   history-supremum PR: "
          + "  ".join(f"{a:.0e}={reached[a] / K:.3f}" for a in (1e-1, 1e-3, 1e-5)))
    rep_pr = sum(
        1 for k in range(len(opts))
        if len(sols) and (np.linalg.norm(sols - opts[k], axis=1) <= rho).any()
        and hf.min() is not None
    )
    print(f"   reported modes within rho of some optimum (any f): "
          f"{rep_pr}/{K}")
