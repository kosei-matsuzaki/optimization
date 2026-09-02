#!/usr/bin/env python3
"""Where does MC-ESO actually stand on peak ratio, function by function?

The goal is to find more global optima than the existing methods on the CEC2013
niching benchmark. Before changing anything, this establishes the starting
position: peak ratio per function, per method, per accuracy level, so the
improvement can be aimed rather than guessed.

Peak ratio scores what a run *reports* (`final_solutions`), never its evaluation
history — a history-based count rewards dense sampling instead of multi-solution
search. See core/runner.niching_peak_metrics.

Budget: the suite's own budgets are 50k-400k evaluations, which is too slow to
iterate on locally, so --evals-frac scales them down. A fraction below 1.0 makes
this an aiming device, not a publishable comparison; the full-budget run belongs
on GitHub Actions.

Usage:
  python3 scripts/niching_baseline.py [--evals-frac 0.1] [--seeds 3]
                                      [--methods MC-ESO,NCDE,...] [--csv out.csv]
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.runner import NICHE_ACCURACIES, _niching_counts           # noqa: E402
from core.optimizers import (MultiChannelEpidemicOptimizer, NCDEOptimizer,
                             RingPSOOptimizer, DEOptimizer,
                             MultistartNelderMeadOptimizer)         # noqa: E402
from core.optimizers.mceso_crowding import MCESOCrowding            # noqa: E402

_METHODS: dict = {
    "MC-ESO": (MultiChannelEpidemicOptimizer, {}),
    "MC-ESO-crowd": (MCESOCrowding, {}),
    "NCDE": (NCDEOptimizer, {}),
    "r3pso": (RingPSOOptimizer, {}),
    "DE": (DEOptimizer, {}),
    "NM-Restart": (MultistartNelderMeadOptimizer, {}),
}
try:                                    # pynmmso breaks on some Python versions
    from core.optimizers import NMMSOOptimizer
    _METHODS["NMMSO"] = (NMMSOOptimizer, {})
except Exception:                                     # pragma: no cover
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", type=str, default="")
    ap.add_argument("--methods", type=str, default="")
    ap.add_argument("--evals-frac", type=float, nargs="*", default=[0.1],
                    help="budget as a fraction of the suite's own, one run per "
                         "value. Several values sweep the budget so peak ratio "
                         "can be read against evaluations per optimum.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--csv", type=Path, default=Path("analysis/niching_baseline.csv"))
    args = ap.parse_args()

    names = ([s.strip() for s in args.funcs.split(",")] if args.funcs
             else sorted(NICHING_BENCHMARKS_BY_NAME))
    methods = ([s.strip() for s in args.methods.split(",")] if args.methods
               else list(_METHODS))
    missing = [m for m in methods if m not in _METHODS]
    if missing:
        raise SystemExit(f"unknown or unavailable methods: {', '.join(missing)}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.csv, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["function", "method", "seed", "evals", "n_optima", "n_reported"]
               + [f"pr_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES])

    acc_lbl = "  ".join(f"{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES)
    print(f"peak ratio on the reported set, {args.seeds} seeds, "
          f"budget = {', '.join(f'{f:g}' for f in args.evals_frac)} x suite")
    if min(args.evals_frac) < 1.0:
        print("(reduced budget: this aims the work, it is not the comparison to "
              "report)")
    print(f"\n{'function':<20}{'method':<14}{'K':>4}{'evals/K':>9}{'|rep|':>7}   {acc_lbl}   "
          f"{'mean':>6}{'s':>6}")
    print("-" * 90)

    for name in names:
        b = NICHING_BENCHMARKS_BY_NAME[name]
        for frac in args.evals_frac:
            budget = max(1000, int(b.suite_max_evals * frac))
            for m in methods:
                cls, kw = _METHODS[m]
                t0 = time.time()
                results = [cls(b, seed=s * 100, **kw).optimize(budget)
                           for s in range(args.seeds)]
                counts, n_rep = _niching_counts(results, b, NICHE_ACCURACIES)
                pr = counts.mean(axis=0) / b.n_global_optima
                for i in range(len(results)):
                    w.writerow([name, m, i, budget, b.n_global_optima, n_rep[i]]
                               + [f"{c / b.n_global_optima:.4f}" for c in counts[i]])
                cells = "  ".join(f"{v:5.2f}" for v in pr)
                print(f"{name:<20}{m:<14}{b.n_global_optima:>4}"
                      f"{budget / b.n_global_optima:>9.0f}{np.mean(n_rep):>7.0f}"
                      f"   {cells}   {pr.mean():>6.3f}{time.time() - t0:>6.0f}")
            print()

    fh.close()
    print(f"rows written to {args.csv}")


if __name__ == "__main__":
    main()
