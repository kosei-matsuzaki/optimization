#!/usr/bin/env python3
"""What is it about a reported set that decides which unmodelled criterion it survives?

The audit found that the winners change completely with the shape of the
unknown: under a linear bias nobody beats reporting the K best local minima,
under instance shifts NCDE and MAP-Elites win, under a forbidden region MC-ESO
and IPOP win. That reads as arbitrary until the sets themselves are measured.

For every (function, method, seed) this records four properties of the set the
method reports, so they can be correlated against its per-model performance:

  n_report   how many solutions it hands back at all
  spread     mean pairwise distance among the top K, as a fraction of the span
  distinct   how many of the top K are further apart than the dedup radius
  quality    mean f of the top K, normalised by the pool's own spread of optima

The hypothesis being tested: spread should predict performance where the unknown
moves the optimum somewhere else (instances, forbidden regions), and quality
should predict it where the unknown only reorders nearby solutions (a bias).
Refuted if neither property correlates with either model (|rho| < 0.3).

Usage:
  python3 scripts/set_properties.py [--funcs ...] [--seeds 3] [--csv out.csv]
"""
from __future__ import annotations
import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import BENCHMARKS_BY_NAME                      # noqa: E402
from scripts.scenario_value import _METHODS                          # noqa: E402


def _pool_spread(f, lo, hi, dim, rng, n_starts=30):
    """Spread of the local optima, the scale the audit normalises everything to."""
    vals = []
    for _ in range(n_starts):
        r = minimize(f, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                     bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
        vals.append(float(r.fun))
    v = np.array(vals)
    return float(np.percentile(v, 75) - v.min()) or 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", type=str, default="")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dedup", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--csv", type=Path, default=Path("analysis/set_properties.csv"))
    args = ap.parse_args()

    names = ([s.strip() for s in args.funcs.split(",")] if args.funcs
             else [n for n in sorted(BENCHMARKS_BY_NAME) if n.startswith("F")])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.csv, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["function", "method", "seed", "n_report", "spread", "distinct",
                "quality"])
    print(f"{'function':<20}{'method':<14}{'n_rep':>7}{'spread':>8}"
          f"{'distinct':>9}{'quality':>9}")
    print("-" * 67)

    for name in names:
        b = BENCHMARKS_BY_NAME[name]
        lo, hi = b.bounds
        span, dim = hi - lo, b.dim

        def f(x):
            return float(b.func(np.clip(np.asarray(x, float), lo, hi)))

        scale = _pool_spread(f, lo, hi, dim, np.random.default_rng(0))
        agg: dict[str, list[tuple[float, ...]]] = {}
        for seed in range(args.seeds):
            for m, (cls, kw) in _METHODS.items():
                kwargs = dict(kw)
                if "sigma0" in cls.__init__.__code__.co_varnames:
                    kwargs.setdefault("sigma0", 0.2 * span)
                res = cls(b, seed=seed * 100, **kwargs).optimize(args.budget)
                rep = np.asarray(res.final_solutions or [res.best_x], dtype=float)
                order = np.argsort([f(x) for x in rep])

                kept: list[np.ndarray] = []
                for i in order:
                    x = rep[i]
                    if all(float(np.linalg.norm(x - y)) > args.dedup * span
                           for y in kept):
                        kept.append(x)
                    if len(kept) == args.k:
                        break
                P = np.array(kept)
                pair = [float(np.linalg.norm(a - c)) / span
                        for a, c in itertools.combinations(P, 2)]
                spread = float(np.mean(pair)) if pair else 0.0
                quality = float(np.mean([f(x) for x in P])) / scale
                row = (len(rep), spread, len(P), quality)
                agg.setdefault(m, []).append(row)
                w.writerow([name, m, seed, len(rep), f"{spread:.5f}", len(P),
                            f"{quality:.6g}"])

        for m, rows in agg.items():
            a = np.mean(np.array(rows, dtype=float), axis=0)
            print(f"{name:<20}{m:<14}{a[0]:>7.0f}{a[1]:>8.3f}{a[2]:>9.1f}"
                  f"{a[3]:>9.3f}")
    fh.close()
    print(f"\nrows written to {args.csv}")


if __name__ == "__main__":
    main()
