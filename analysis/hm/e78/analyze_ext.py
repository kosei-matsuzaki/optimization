#!/usr/bin/env python3
"""Stage 2: are the four discordant functions of the 20-seed gate real?

Stage 1 (seeds 0-19, all 24 functions) left SR@1e-10 numerically unchanged at
0.9208 -- but not by identity: 4 cells were won and 4 lost, on four named
functions. 4 vs 4 is p = 1.0 under an exact sign test, so stage 1 cannot say
whether the arm swaps cells at random or trades a real loss on one function for
a real gain on another. Entry 41's rule applies: the pre-registered judgement is
settled on the pre-registered seeds, and the extension is a *separate* stage
that only resolves the null.

Stage 2 therefore re-runs the four functions stage 1 named, on **fresh seeds
20-59**, so the seeds that selected the functions are not the seeds that test
them. Reported per function: SR@1e-10 on both arms and the exact two-sided sign
test on the discordant cells. 40 fresh seeds put the sign test's floor at
2/2**n for the discordant count, so a one-sided-looking 5:0 split is already
p = 0.0625 and 6:0 is p = 0.031 -- enough to separate "real regression" from
"seed noise" if the effect is as large as stage 1 suggested (3/20 on F04).

Usage: python3 analysis/hm/e78/analyze_ext.py analysis/hm/e78/ext5k_*.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import comb

import numpy as np

#  Stage 1's per-function discordant counts (seeds 0-19), for the side-by-side.
STAGE1 = {"F04-BucheRastrigin": (0, 3), "F20-Schwefel": (0, 1),
          "F17-SchafferF7": (3, 0), "F23-Katsuura": (1, 0)}


def sign_test(won: int, lost: int) -> float:
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    by = defaultdict(list)
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                by[r["function"]].append(r)

    print(f"{'function':<24}{'n':>4}{'SRb':>7}{'SRv':>7}{'won':>5}{'lost':>6}"
          f"{'p':>9}   {'stage1 (w/l)':>13}   pooled p")
    print("-" * 92)
    for f in sorted(by):
        rs = by[f]
        b = np.array([float(r["base_best_f"]) <= 1e-10 for r in rs])
        v = np.array([float(r["var_best_f"]) <= 1e-10 for r in rs])
        w = int((v & ~b).sum())
        l = int((~v & b).sum())
        w1, l1 = STAGE1.get(f, (0, 0))
        print(f"{f:<24}{len(rs):>4}{b.mean():>7.3f}{v.mean():>7.3f}{w:>5}{l:>6}"
              f"{sign_test(w, l):>9.4f}   {f'{w1}/{l1}':>13}   "
              f"{sign_test(w + w1, l + l1):.4f}")
    print("\n  pooled p combines stage 1 and stage 2 cells; it is reported for "
          "completeness only.\n  Stage 2's own p is the one that is free of the "
          "selection that picked these four functions.")


if __name__ == "__main__":
    main()
