#!/usr/bin/env python3
"""Do the sets that real niching/QD methods report pay off under unmodelled criteria?

scripts/complementarity_test.py answered the question for a *stand-in* for
geometric diversity (farthest-point selection from a multistart pool). The
obvious objection is that no one runs that rule — so this scores the sets that
the actual methods report.

Each method optimises the nominal objective f with a fixed budget and hands back
its reported solution set (OptimizeResult.final_solutions — the same set the
CEC2013 peak-ratio metric scores). The set is truncated to its K best by nominal
f, which is what a practitioner presenting K options would do. Then the set is
scored on held-out scenarios q(x) = f(x) + eps * (w . x), which no method ever
saw: the unmodelled criterion the whole diverse-solutions literature invokes as
its justification.

Two reference rules are computed from a multistart pool of the same budget:

  quality     the K best local minima by nominal f — no diversity at all
  complement  K chosen greedily to minimise regret on TRAINING scenarios; this
              one sees scenarios, so it is an upper reference, not a competitor

Usage:
  python3 scripts/scenario_value.py [--funcs ...] [--budget 5000] [--k 5]
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import BENCHMARKS_BY_NAME                      # noqa: E402
from core.optimizers import (MultiChannelEpidemicOptimizer,          # noqa: E402
                             NCDEOptimizer, RingPSOOptimizer, NMMSOOptimizer,
                             MultistartNelderMeadOptimizer,
                             IPOPCMAESOptimizer, DEOptimizer, PSOOptimizer)

_METHODS = {
    "MC-ESO":     (MultiChannelEpidemicOptimizer, {}),
    "NMMSO":      (NMMSOOptimizer, {}),
    "NCDE":       (NCDEOptimizer, {}),
    "r3pso":      (RingPSOOptimizer, {}),
    "NM-Restart": (MultistartNelderMeadOptimizer, {}),
    "IPOP-CMA-ES": (IPOPCMAESOptimizer, {}),
    "DE":         (DEOptimizer, {}),
    # Two controlled contrasts for the attraction hypothesis: methods whose
    # population is pulled toward an incumbent lose ground as the unmodelled
    # criterion grows, methods without such a pull do not.
    #   PSO vs r3pso      global best vs ring neighbourhood best (topology)
    #   DE vs Crowding-DE parent replacement vs nearest replacement, same mutation
    "PSO":        (PSOOptimizer, {}),
    "Crowding-DE": (NCDEOptimizer, {"m": 30}),
}
_FUNCS = ["F03-RastriginSep", "F15-RastriginRot", "F21-Gallagher101", "F17-SchafferF7"]


def _multistart_pool(f, lo, hi, dim, rng, n_starts, tol):
    found: list[np.ndarray] = []
    for _ in range(n_starts):
        r = minimize(f, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                     bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
        x = np.clip(r.x, lo, hi)
        if all(float(np.linalg.norm(x - y)) > tol for y in found):
            found.append(x)
    return np.array(found)


def _greedy_complement(vals, opt, k):
    chosen, cur = [], np.full(vals.shape[1], np.inf)
    for _ in range(min(k, len(vals))):
        best_i, best_score = None, np.inf
        for i in range(len(vals)):
            if i in chosen:
                continue
            score = float(np.mean(np.minimum(cur, vals[i]) - opt))
            if score < best_score:
                best_i, best_score = i, score
        chosen.append(best_i)
        cur = np.minimum(cur, vals[best_i])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", type=str, default=",".join(_FUNCS))
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tilt", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dedup", type=float, default=0.05,
                    help="drop reported points within this fraction of the span of "
                         "a better one before taking the top K. Without it a "
                         "converged population reports K near-duplicates, which is "
                         "not what anyone would present.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write one row per (function, method, seed, scenario) so "
                         "the comparison can be tested pairwise instead of on means")
    ap.add_argument("--raw", action="store_true",
                    help="also score the un-deduplicated top-K, to show what the "
                         "duplicates cost")
    args = ap.parse_args()

    writer = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["function", "method", "seed", "scenario", "regret"])
    print(f"reported sets under an unmodelled criterion   budget={args.budget}  "
          f"K={args.k}  tilt={args.tilt}  train/test={args.n_train}/{args.n_test}  "
          f"seeds={args.seeds}")
    head = (f"{'function':<20}{'method':<13}{'|report|':>9}{'regret':>10}"
            f"{'vs quality':>12}")
    print(head)
    print("-" * len(head))

    for name in [s.strip() for s in args.funcs.split(",")]:
        b = BENCHMARKS_BY_NAME[name]
        lo, hi = b.bounds
        span, dim = hi - lo, b.dim

        def f(x):
            return float(b.func(np.clip(np.asarray(x, float), lo, hi)))

        rows: dict[str, list[float]] = {}
        sizes: dict[str, list[int]] = {}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            pool = _multistart_pool(f, lo, hi, dim, rng, 50, 0.01 * span)
            f_pool = np.array([f(x) for x in pool])
            spread = float(np.percentile(f_pool, 75) - f_pool.min()) or 1.0
            eps = args.tilt * spread / (span * dim ** 0.5)

            n_all = args.n_train + args.n_test
            W = rng.normal(size=(n_all, dim))
            W /= np.linalg.norm(W, axis=1, keepdims=True)

            def q(x, s):
                return f(x) + eps * float(np.dot(W[s], np.asarray(x, float)))

            def score(points):
                """Mean held-out regret of the best-of-set, plus its own values."""
                P = np.asarray(points, dtype=float)
                v = np.array([[q(x, s) for s in range(n_all)] for x in P])
                return v

            pool_vals = score(pool)

            # Reference per scenario: best of everything evaluated so far, filled
            # in again below once each method's points are known.
            best_known = pool_vals.min(axis=0)

            sets: dict[str, np.ndarray] = {}
            for m, (cls, kw) in _METHODS.items():
                kwargs = dict(kw)
                if cls is IPOPCMAESOptimizer:
                    kwargs["sigma0"] = 0.2 * span
                res = cls(b, seed=seed * 100, **kwargs).optimize(args.budget)
                rep = np.asarray(res.final_solutions or [res.best_x], dtype=float)
                order = np.argsort([f(x) for x in rep])
                if args.raw:
                    sets[m + " (raw)"] = rep[order[:args.k]]
                # Deduplicate the way a peak-ratio count would: walk in quality
                # order and keep a point only if it is far from every better one.
                kept: list[np.ndarray] = []
                for i in order:
                    x = rep[i]
                    if all(float(np.linalg.norm(x - y)) > args.dedup * span
                           for y in kept):
                        kept.append(x)
                    if len(kept) == args.k:
                        break
                sets[m] = np.array(kept)
                sizes.setdefault(m, []).append(len(rep))

            # Reference rules from the multistart pool, same budget in spirit.
            sets["quality"] = pool[np.argsort(f_pool)[:args.k]]
            tr = slice(0, args.n_train)
            idx = _greedy_complement(pool_vals[:, tr], pool_vals[:, tr].min(axis=0),
                                     args.k)
            sets["complement"] = pool[idx]

            vals = {m: score(P) for m, P in sets.items()}
            for v in vals.values():
                best_known = np.minimum(best_known, v.min(axis=0))
            te = slice(args.n_train, n_all)
            for m, v in vals.items():
                per_scen = np.min(v[:, te], axis=0) - best_known[te]
                rows.setdefault(m, []).append(float(np.mean(per_scen)))
                if writer is not None:
                    for j, val in enumerate(per_scen):
                        writer.writerow([name, m, seed, args.n_train + j,
                                         f"{float(val):.6g}"])

        base = float(np.mean(rows["quality"]))
        order = sorted(rows, key=lambda m: float(np.mean(rows[m])))
        for m in order:
            r = float(np.mean(rows[m]))
            n_rep = int(np.mean(sizes[m.replace(" (raw)", "")]))                 if m.replace(" (raw)", "") in sizes else args.k
            ratio = r / base if base > 0 else float("nan")
            print(f"{name:<20}{m:<13}{n_rep:>9}{r:>10.3f}{ratio:>11.2f}x")
        print()

    if writer is not None:
        fh.close()
        print(f"per-scenario rows written to {args.csv}")
    print("regret: mean over held-out scenarios of the best-of-K, against the best "
          "value anything reached (lower is better).")
    print("vs quality: ratio to simply reporting the K best local minima. >1 means "
          "the method's reported set is worse than not diversifying at all.")
    print("'complement' sees training scenarios, so it is an upper reference.")


if __name__ == "__main__":
    main()
