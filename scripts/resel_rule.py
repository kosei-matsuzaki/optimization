#!/usr/bin/env python3
"""Does the history-reselection reporting rule change the ranking, or just lift
everybody at the loosest accuracy?

Entry 20 measured the rho-greedy reselection (`reselect_from_history`, zero
extra evaluations, cap = max(100, 2K)) on MC-ESO only, where it took N07/N09
PR@1e-1 from 0.25/0.12 to 1.00. That number cannot be compared against NMMSO,
because the rule is method-agnostic post-processing: every method can adopt it,
so the honest comparison gives it to all of them. `niching_baseline.py
--report-rule both` does that, scoring both rules off the *same* runs.

This reads that CSV and answers the two questions the comparison was set up for:

  (a) does the ranking move?  Per function and accuracy, methods are ranked by
      mean PR under each rule and the two orders are printed side by side.
  (b) is the gain concentrated at eps=1e-1 for every method, not just MC-ESO?
      Per method and accuracy, the paired per-seed gain (reselect - current)
      with w/t/l, Wilcoxon p and A12. Both rules score the same runs, so the
      pairing is exact and the test has no run-to-run noise in it.

If (b) holds across methods, PR@1e-1 does not separate methods once reporting is
equalised, and later judgments belong at 1e-3 or below.

Usage:
  python3 scripts/resel_rule.py analysis/hm/resel_2d.csv analysis/hm/resel_n08.csv ...
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hunt_confound import paired                            # noqa: E402

EPS_COLS = ["pr_1e-1", "pr_1e-2", "pr_1e-3", "pr_1e-4", "pr_1e-5"]


def load(paths: list[str]) -> dict:
    """{(function, method, rule, eps_col): {seed: pr}} plus the reported-set size."""
    pr: dict = defaultdict(dict)
    nrep: dict = defaultdict(dict)
    funcs: list[str] = []
    methods: list[str] = []
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                f, m, rule, s = r["function"], r["method"], r["rule"], int(r["seed"])
                if f not in funcs:
                    funcs.append(f)
                if m not in methods:
                    methods.append(m)
                nrep[(f, m, rule)][s] = float(r["n_reported"])
                for c in EPS_COLS:
                    pr[(f, m, rule, c)][s] = float(r[c])
    return pr, nrep, funcs, methods


def vec(d: dict, key) -> np.ndarray:
    """Per-seed vector in seed order; empty if the cell was never measured."""
    cell = d.get(key)
    if not cell:
        return np.zeros(0)
    return np.array([cell[s] for s in sorted(cell)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    args = ap.parse_args()
    pr, nrep, funcs, methods = load(args.csvs)

    # ---- (b) per-method gain, accuracy by accuracy -------------------------
    print("=" * 100)
    print("(b) paired gain of the reselection rule (reselect - current), same runs")
    print("=" * 100)
    for f in funcs:
        print(f"\n{f}")
        print(f"{'method':<14}{'|rep| cur':>10}{'|rep| res':>10}   "
              + "".join(f"{c.replace('pr_', ''):>22}" for c in EPS_COLS))
        for m in methods:
            cur_n, res_n = vec(nrep, (f, m, "current")), vec(nrep, (f, m, "reselect"))
            if len(cur_n) == 0 or len(res_n) == 0:
                continue
            cells = []
            for c in EPS_COLS:
                x, y = vec(pr, (f, m, "reselect", c)), vec(pr, (f, m, "current", c))
                wtl, p, _ = paired(x, y)
                cells.append(f"{y.mean():.2f}->{x.mean():.2f} {wtl:>7}")
            print(f"{m:<14}{cur_n.mean():>10.0f}{res_n.mean():>10.0f}   "
                  + "".join(f"{c:>22}" for c in cells))

    # ---- (a) does the ranking move? ---------------------------------------
    print("\n" + "=" * 100)
    print("(a) ranking under each rule (mean PR over seeds, best first)")
    print("=" * 100)
    for f in funcs:
        print(f"\n{f}")
        for c in EPS_COLS:
            order = {}
            for rule in ("current", "reselect"):
                got = [(vec(pr, (f, m, rule, c)).mean(), m) for m in methods
                       if len(vec(pr, (f, m, rule, c)))]
                order[rule] = sorted(got, key=lambda t: -t[0])
            same = [m for _, m in order["current"]] == [m for _, m in order["reselect"]]
            lbl = c.replace("pr_", "")
            for rule in ("current", "reselect"):
                cells = "  ".join(f"{m}:{v:.2f}" for v, m in order[rule])
                tag = lbl if rule == "current" else ""
                print(f"  {tag:>6} {rule:<9} {cells}")
            print(f"  {'':>6} {'order':<9} "
                  f"{'UNCHANGED' if same else '*** CHANGED ***'}")

    # ---- summary of (b): is the gain 1e-1-only, across methods? -----------
    print("\n" + "=" * 100)
    print("(b) summary: mean gain per accuracy, pooled over functions")
    print("=" * 100)
    print(f"{'method':<14}" + "".join(f"{c.replace('pr_', ''):>10}" for c in EPS_COLS))
    for m in methods:
        row = []
        for c in EPS_COLS:
            g = [vec(pr, (f, m, "reselect", c)).mean() - vec(pr, (f, m, "current", c)).mean()
                 for f in funcs if len(vec(pr, (f, m, "current", c)))]
            row.append(np.mean(g) if g else float("nan"))
        print(f"{m:<14}" + "".join(f"{v:>+10.3f}" for v in row))


if __name__ == "__main__":
    main()
