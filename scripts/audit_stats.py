#!/usr/bin/env python3
"""Paired tests for the diverse-solutions audit.

scripts/scenario_value.py reports means, and a mean over 30 scenarios hides
whether a difference is real: an earlier single-seed run had NM-Restart best on
F03-RastriginSep and three seeds put it last. This runs the comparison the way
the rest of the project runs comparisons — paired, per function, with an effect
size — on the per-(seed, scenario) rows that script can now emit.

Each method is compared against `quality`, the baseline that reports the K best
local minima with no diversity mechanism at all, on exactly the same scenarios:

  p        two-sided Wilcoxon signed-rank over the paired regrets
  A12      Vargha-Delaney, share of pairs where the method is better
           (>0.5 favours the method, <0.5 favours plain quality)
  seeds    how many of the seeds agree with the overall direction

Caveat, stated rather than hidden: scenarios drawn within one seed share a tilt
distribution and a function, so they are not independent samples. The p values
are therefore optimistic. The seed-agreement column is the honest robustness
check — a difference that only shows up in some seeds is not a difference.

Usage:
  python3 scripts/audit_stats.py <rows.csv> [--baseline quality] [--alpha 0.05]
"""
from __future__ import annotations
import argparse
import collections
import csv
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAS_SCIPY = True
except ImportError:                                    # pragma: no cover
    _HAS_SCIPY = False


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    """P(a < b) + 0.5 P(a == b) — both are regrets, so lower is better."""
    less = float(np.sum(a < b))
    ties = float(np.sum(a == b))
    return (less + 0.5 * ties) / len(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--baseline", default="quality")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    data: dict[tuple[str, str], dict[tuple[int, int], float]] = collections.defaultdict(dict)
    for r in csv.DictReader(open(args.rows, newline="")):
        data[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
            float(r["regret"])
    funcs = sorted({fn for fn, _ in data})
    methods = sorted({m for _, m in data} - {args.baseline})

    print(f"paired against '{args.baseline}'   functions={len(funcs)}   alpha={args.alpha}")
    print(f"{'method':<14}{'better':>8}{'worse':>7}{'n.s.':>6}{'med A12':>9}"
          f"{'seed-consistent':>17}")
    print("-" * 61)

    detail: dict[str, list[tuple[str, float, float, int, int]]] = {}
    for m in methods:
        better = worse = ns = 0
        a12s, consistent, total = [], 0, 0
        rows = []
        for fn in funcs:
            base = data.get((fn, args.baseline), {})
            cand = data.get((fn, m), {})
            keys = sorted(set(base) & set(cand))
            if len(keys) < 10:
                continue
            a = np.array([cand[k] for k in keys])
            b = np.array([base[k] for k in keys])
            a12 = _a12(a, b)
            a12s.append(a12)
            p = float("nan")
            if _HAS_SCIPY and np.any(a != b):
                try:
                    p = float(_wilcoxon(a, b, zero_method="wilcox").pvalue)
                except ValueError:
                    p = float("nan")
            sig = (p == p) and p < args.alpha
            if sig and a12 > 0.5:
                better += 1
            elif sig and a12 < 0.5:
                worse += 1
            else:
                ns += 1
            # Does every seed point the same way as the pooled result?
            seeds = sorted({s for s, _ in keys})
            signs = []
            for s in seeds:
                ks = [k for k in keys if k[0] == s]
                signs.append(np.mean([cand[k] for k in ks])
                             < np.mean([base[k] for k in ks]))
            agree = sum(1 for x in signs if x == (a12 > 0.5))
            consistent += agree
            total += len(signs)
            rows.append((fn, a12, p, agree, len(signs)))
        detail[m] = rows
        share = f"{consistent}/{total}" if total else "-"
        print(f"{m:<14}{better:>8}{worse:>7}{ns:>6}"
              f"{np.median(a12s) if a12s else float('nan'):>9.2f}{share:>17}")

    print("\nper function (A12 against the baseline; * = p < alpha; "
          "seeds agreeing with the direction)")
    for m in methods:
        line = []
        for fn, a12, p, agree, n in detail[m]:
            star = "*" if (p == p and p < args.alpha) else " "
            line.append(f"{fn.split('-')[0]}:{a12:.2f}{star}{agree}/{n}")
        print(f"  {m:<13}" + "  ".join(line))
    print("\nA12 > 0.5 means the method's reported set beats reporting the best K "
          "local minima on that share of paired scenarios.")
    print("Scenarios within a seed are not independent, so p is optimistic; the "
          "seed-agreement counts are the robustness check.")


if __name__ == "__main__":
    main()
