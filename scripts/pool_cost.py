#!/usr/bin/env python3
"""What does the reference pool cost, next to the budget the methods get?

Every reference rule in the audit -- reporting the K best, the greedy
complement, the distance-within-tolerance rule -- selects from a pool built by
50 L-BFGS-B multistarts. The methods it is compared against get --budget
evaluations, 5000 by default. If the pool costs far more than that, then the
audit's headline (that a reported set does not beat reporting the K best local
minima) is partly a statement about budget rather than about diversity.

This counts the evaluations the pool actually consumes, per function, and
prints it next to the methods' budget.

Usage:
  python3 scripts/pool_cost.py [--budget 5000] [--seeds 3]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import BENCHMARKS_BY_NAME                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", type=str, default="")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    names = ([s.strip() for s in args.funcs.split(",")] if args.funcs
             else [n for n in sorted(BENCHMARKS_BY_NAME) if n.startswith("F")])

    print(f"pool = {args.starts} L-BFGS-B multistarts, maxfun 2000 each; "
          f"methods get {args.budget}")
    print(f"{'function':<22}{'pool evals':>12}{'x budget':>10}{'n kept':>8}")
    print("-" * 52)
    ratios = []
    for name in names:
        b = BENCHMARKS_BY_NAME[name]
        lo, hi = b.bounds
        span, dim = hi - lo, b.dim
        totals, kept_n = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            n = 0

            def f(x):
                nonlocal n
                n += 1
                return float(b.func(np.clip(np.asarray(x, float), lo, hi)))

            found: list[np.ndarray] = []
            for _ in range(args.starts):
                r = minimize(f, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                             bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
                x = np.clip(r.x, lo, hi)
                if all(float(np.linalg.norm(x - y)) > 0.01 * span for y in found):
                    found.append(x)
            totals.append(n)
            kept_n.append(len(found))
        m = float(np.mean(totals))
        ratios.append(m / args.budget)
        print(f"{name:<22}{m:>12.0f}{m / args.budget:>10.2f}{np.mean(kept_n):>8.1f}")

    r = np.array(ratios)
    print(f"\nmedian {np.median(r):.2f}x the budget, max {r.max():.2f}x, "
          f"{100 * np.mean(r > 1):.0f}% of functions above 1x")
    print("Gradients are counted too: L-BFGS-B without an analytic gradient spends")
    print("dim extra evaluations per finite-difference gradient, which is where")
    print("most of this goes in low dimension.")


if __name__ == "__main__":
    main()
