#!/usr/bin/env python3
"""Where does MC-ESO actually stand on peak ratio, function by function?

The goal is to find more global optima than the existing methods on the CEC2013
niching benchmark. Before changing anything, this establishes the starting
position: peak ratio per function, per method, per accuracy level, so the
improvement can be aimed rather than guessed.

Peak ratio scores what a run *reports* (`final_solutions`), never its evaluation
history — a history-based count rewards dense sampling instead of multi-solution
search. See core/runner.niching_peak_metrics.

``--report-rule`` swaps that reporting rule for a *method-agnostic* one. The
``reselect`` rule rebuilds the reported set from the run's own evaluation
history with the same rho-greedy walk the scorer uses, capped at the
competition's own ``max(100, 2K)`` (scripts.diagnose_niching.reselect_from_history).
It is legal output post-processing that costs zero extra evaluations, so any
method can adopt it — which is exactly why measuring it on one method only
(MC-ESO, 2026-09-02 log entry 20) cannot support a ranking claim. ``both``
scores each run under both rules off the *same* runs, so the comparison is
paired and costs nothing beyond the single set of optimizer runs.

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
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.runner import NICHE_ACCURACIES, _niching_counts           # noqa: E402
from scripts.diagnose_niching import reselect_from_history          # noqa: E402
from core.optimizers import (MultiChannelEpidemicOptimizer, NCDEOptimizer,
                             RingPSOOptimizer, DEOptimizer,
                             MultistartNelderMeadOptimizer)         # noqa: E402
from core.optimizers.mceso_crowding import MCESOCrowding            # noqa: E402
from core.optimizers.mceso_rel_level import RelLevelMCESO           # noqa: E402

_METHODS: dict = {
    "MC-ESO": (MultiChannelEpidemicOptimizer, {}),
    # The adoption candidate: the hunt-release level made relative to the
    # scoring accuracy, L = c * eps_target with c = 1.0 (entry 46 located the
    # corner there) and the endpoint clamp on f_init_scale (entry 44). This is
    # a *diagnostic arm*, not a default change -- core/optimizers/mceso.py is
    # untouched. It exists here so the candidate can be scored by the same
    # driver, at the same budget, as the baselines it is compared against.
    "MC-ESO-rel": (RelLevelMCESO, {"rel_level": 1e-5, "fis_floor": 1e-12}),
    # Same rule with the *design* target moved one decade down: L = eps_target
    # with eps_target read as 1e-6 instead of 1e-5. Entry 51 left exactly one
    # step in the N08 profile (1e-4 -> 1e-5, 0.860 -> 0.754) and the candidate
    # sits with L *on* the deepest scored threshold, so a basin released at the
    # boundary need not score at 1e-5. This arm tests that reading; it is not a
    # re-sweep of c (entry 46 closed that) -- c stays 1.0, only the absolute
    # level L moves. Diagnostic arm; mceso.py defaults are untouched.
    "MC-ESO-rel6": (RelLevelMCESO, {"rel_level": 1e-6, "fis_floor": 1e-12}),
    # Entry 75 (question 1, second half): the *sigma* release clause, on top of
    # the same c = 1.0 arm -- which entry 74 showed is bit-identical to base on
    # N18-CF3-10D, so on that function these two are "base + one sigma knob".
    # Entry 55 closed both for free on N08-Shubert3D; entry 75's firing probe
    # says which one can bite on N18: at release sigma sits *at* its floor
    # (median 1.00 floor units) in 12042 releases per run, so `_sig10` (tol
    # 1.5 -> 1.0) can only touch the 6% released above the floor, while `_fl08`
    # (floor 1e-6 -> 1e-8) gives 99.7% of them two more decades to drill.
    # Diagnostic arms; core/optimizers/mceso.py defaults are untouched.
    "MC-ESO-rel-sig10": (RelLevelMCESO, {"rel_level": 1e-5, "fis_floor": 1e-12,
                                         "exhausted_sigma_tol": 1.0}),
    "MC-ESO-rel-fl08": (RelLevelMCESO, {"rel_level": 1e-5, "fis_floor": 1e-12,
                                        "sigma_floor_ratio": 1e-8}),
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


def _reselected_results(results, b):
    """Same runs, reported set replaced by the rho-greedy pick from the run's own
    history. Returns fresh OptimizeResult copies so the original reported sets
    stay intact and both rules can be scored off one set of runs."""
    cap = max(100, 2 * b.n_global_optima)          # core.runner._niching_counts
    out = []
    for r in results:
        hx = np.asarray(r.history_x, dtype=float)
        hf = np.asarray(r.history_f, dtype=float)
        rx, _ = reselect_from_history(hx, hf, b.niche_rho, cap)
        out.append(dataclasses.replace(r, final_solutions=[x.copy() for x in rx]))
    return out


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
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first seed index (default 0). Seeds stay numbered "
                         "globally -- seed index i always means optimiser seed "
                         "i*100 and is written as `seed=i` -- so one function "
                         "can be sharded across processes and the shards "
                         "concatenated, or paired seed-by-seed against a stored "
                         "CSV from an earlier cycle.")
    ap.add_argument("--report-rule", type=str, default="current",
                    choices=("current", "reselect", "both"),
                    help="which reporting rule scores the runs. 'current' = the "
                         "method's own final_solutions (the historical table). "
                         "'reselect' = rho-greedy pick from the run's own "
                         "history, capped at max(100, 2K), zero extra "
                         "evaluations. 'both' scores each run under both, "
                         "paired, off the same runs.")
    ap.add_argument("--csv", type=Path, default=Path("analysis/hm/niching_baseline.csv"))
    args = ap.parse_args()
    rules = (("current", "reselect") if args.report_rule == "both"
             else (args.report_rule,))

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
    w.writerow(["function", "method", "rule", "seed", "evals", "n_optima",
                "n_reported"]
               + [f"pr_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES])

    acc_lbl = "  ".join(f"{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES)
    print(f"peak ratio on the reported set, {args.seeds} seeds, "
          f"budget = {', '.join(f'{f:g}' for f in args.evals_frac)} x suite")
    if min(args.evals_frac) < 1.0:
        print("(reduced budget: this aims the work, it is not the comparison to "
              "report)")
    if len(rules) > 1:
        print("(both reporting rules are scored off the same runs: paired, no "
              "extra evaluations)")
    print(f"\n{'function':<20}{'method':<14}{'rule':<9}{'K':>4}{'evals/K':>9}"
          f"{'|rep|':>7}   {acc_lbl}   {'mean':>6}{'s':>6}")
    print("-" * 99)

    for name in names:
        b = NICHING_BENCHMARKS_BY_NAME[name]
        for frac in args.evals_frac:
            budget = max(1000, int(b.suite_max_evals * frac))
            for m in methods:
                cls, kw = _METHODS[m]
                t0 = time.time()
                seeds = range(args.seed_offset, args.seed_offset + args.seeds)
                results = [cls(b, seed=s * 100, **kw).optimize(budget)
                           for s in seeds]
                for rule in rules:
                    scored = (results if rule == "current"
                              else _reselected_results(results, b))
                    counts, n_rep = _niching_counts(scored, b, NICHE_ACCURACIES)
                    pr = counts.mean(axis=0) / b.n_global_optima
                    for i in range(len(scored)):
                        w.writerow([name, m, rule, args.seed_offset + i, budget,
                                    b.n_global_optima, n_rep[i]]
                                   + [f"{c / b.n_global_optima:.4f}"
                                      for c in counts[i]])
                    cells = "  ".join(f"{v:5.2f}" for v in pr)
                    print(f"{name:<20}{m:<14}{rule:<9}{b.n_global_optima:>4}"
                          f"{budget / b.n_global_optima:>9.0f}"
                          f"{np.mean(n_rep):>7.0f}"
                          f"   {cells}   {pr.mean():>6.3f}"
                          f"{time.time() - t0:>6.0f}")
                fh.flush()
            print()

    fh.close()
    print(f"rows written to {args.csv}")


if __name__ == "__main__":
    main()
