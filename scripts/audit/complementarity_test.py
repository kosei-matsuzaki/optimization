#!/usr/bin/env python3
"""Kill test: is geometric diversity a good proxy for complementarity?

Every diverse-solutions method in use — niching, QD, EDO — decides what counts as
"another solution" from the geometry of the search space (a radius, a behaviour
descriptor, a distance). The reason a practitioner wants several solutions is
usually different: the objective they optimise is a proxy, and the real criterion
is revealed later. What matters then is that *one member of the reported set*
turns out to be good under whichever criterion shows up. Those two are not the
same thing, and the gap does not appear to have been measured.

Scenario model: the true objective is f(x) + eps * (w . x), with an unknown tilt
w drawn per scenario. A tilt reorders which local optimum is best, so covering
the scenario space is not the same as spreading points around the domain, and it
does not collapse into a two-objective trade-off.

The comparison holds the candidate pool fixed (multistart local minima) and
varies only the SELECTION rule, so the result is about geometry vs
complementarity rather than about who searches better:

  quality     the K best candidates by nominal f (no diversity at all)
  geometric   K candidates spread by distance, greedily (niching-style)
  random      K random candidates (control)
  complement  K candidates chosen greedily to minimise mean regret on TRAINING
              scenarios (the K-adaptability / ROMS objective)

Scored by mean regret on held-out scenarios. The reference per scenario is the
best value anything reached — offline multistart *and* the candidate pool —
because an offline optimiser weaker than the pool would make regrets negative.

Usage:
  python3 scripts/complementarity_test.py [--dims 2,5] [--seeds 3]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.benchmarks import (BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME,   # noqa: E402
                             BENCHMARKS_5D_BY_NAME, BENCHMARKS_10D_BY_NAME)

_REGISTRY = {2: BENCHMARKS_BY_NAME, 3: BENCHMARKS_3D_BY_NAME,
             5: BENCHMARKS_5D_BY_NAME, 10: BENCHMARKS_10D_BY_NAME}
_FUNCS = ["F03-RastriginSep", "F15-RastriginRot", "F21-Gallagher101", "F17-SchafferF7"]
_RULES = ["quality", "geometric", "random", "complement"]


def _anchors(f, lo, hi, dim, n_starts, rng, tol):
    """Candidate pool: distinct local minima of the nominal objective."""
    found: list[np.ndarray] = []
    for _ in range(n_starts):
        r = minimize(f, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                     bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
        x = np.clip(r.x, lo, hi)
        if all(float(np.linalg.norm(x - y)) > tol for y in found):
            found.append(x)
    return np.array(found)


def _scenario_opt(q, lo, hi, dim, rng, n_starts):
    best = np.inf
    for _ in range(n_starts):
        r = minimize(q, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                     bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
        best = min(best, float(r.fun))
    return best


def _greedy_complement(vals: np.ndarray, opt: np.ndarray, k: int) -> list[int]:
    """Greedy set minimising mean regret of the best-of-set, per scenario.

    vals[i, s] = value of candidate i in scenario s. Mean-of-min improves with
    diminishing returns as the set grows, so greedy is the natural algorithm.
    """
    chosen: list[int] = []
    cur = np.full(vals.shape[1], np.inf)
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


def _greedy_spread(pts: np.ndarray, f_vals: np.ndarray, k: int) -> list[int]:
    """Niching-style selection: best point first, then farthest-from-chosen."""
    chosen = [int(np.argmin(f_vals))]
    while len(chosen) < min(k, len(pts)):
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[None, chosen, :], axis=2), axis=1)
        d[chosen] = -np.inf
        chosen.append(int(np.argmax(d)))
    return chosen


def _one_seed(b, dim: int, seed: int, args) -> tuple[dict[str, list[float]], int]:
    lo, hi = b.bounds
    span = hi - lo

    def f(x):
        return float(b.func(np.clip(np.asarray(x, float), lo, hi)))

    rng = np.random.default_rng(seed)
    pool = _anchors(f, lo, hi, dim, args.n_starts, rng, tol=0.01 * span)
    if len(pool) < 2:
        return {}, len(pool)
    f_pool = np.array([f(x) for x in pool])

    # A tilt worth studying is one that can reorder the optima, so scale it to
    # the spread of the pool's values rather than to the function's range.
    spread = float(np.percentile(f_pool, 75) - f_pool.min()) or 1.0
    eps = args.tilt * spread / (span * dim ** 0.5)

    n_all = args.n_train + args.n_test
    W = rng.normal(size=(n_all, dim))
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    def q_of(s):
        w = W[s]
        return lambda x: f(x) + eps * float(np.dot(w, np.asarray(x, float)))

    vals = np.array([[q_of(s)(x) for s in range(n_all)] for x in pool])
    opt = np.minimum(
        np.array([_scenario_opt(q_of(s), lo, hi, dim, rng, 20) for s in range(n_all)]),
        vals.min(axis=0))
    tr, te = slice(0, args.n_train), slice(args.n_train, n_all)

    pick = {
        "quality":    lambda k: list(np.argsort(f_pool)[:k]),
        "geometric":  lambda k: _greedy_spread(pool, f_pool, k),
        "random":     lambda k: list(rng.choice(len(pool), size=min(k, len(pool)),
                                                replace=False)),
        "complement": lambda k: _greedy_complement(vals[:, tr], opt[tr], k),
    }
    out: dict[str, list[float]] = {}
    for rule in _RULES:
        row = []
        for k in args.ks:
            idx = pick[rule](min(k, len(pool)))
            row.append(float(np.mean(np.min(vals[idx][:, te], axis=0) - opt[te])))
        out[rule] = row
    return out, len(pool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=str, default="2,5")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-starts", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--tilt", type=float, default=0.5,
                    help="tilt size as a fraction of the spread of local optimum "
                         "values; 0.5 lets a tilt reorder optima that differ by "
                         "half that spread")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    args.ks = [1, 2, 3, 5, args.k_max]

    print(f"geometry vs complementarity   starts={args.n_starts}  "
          f"train/test scenarios={args.n_train}/{args.n_test}  tilt={args.tilt}  "
          f"seeds={args.seeds}")
    header = (f"{'function':<20}{'dim':>4}{'cand':>6}  {'rule':<11}"
              + "".join(f"{f'K={k}':>10}" for k in args.ks))
    print(header)
    print("-" * len(header))

    for name in _FUNCS:
        for dim in [int(s) for s in args.dims.split(",")]:
            reg = _REGISTRY.get(dim)
            if reg is None or name not in reg:
                continue
            acc: dict[str, list[list[float]]] = {r: [] for r in _RULES}
            pools = []
            for seed in range(args.seeds):
                res, n_pool = _one_seed(reg[name], dim, seed, args)
                pools.append(n_pool)
                for rule, row in res.items():
                    acc[rule].append(row)
            if not acc["quality"]:
                continue
            for rule in _RULES:
                mean = np.mean(np.array(acc[rule]), axis=0)
                print(f"{name:<20}{dim:>4}{int(np.mean(pools)):>6}  {rule:<11}"
                      + "".join(f"{v:>10.3f}" for v in mean))
            print()
    print("cells: mean regret on held-out scenarios, averaged over seeds "
          "(lower is better).")
    print("'geometric' tracking 'complement' would mean geometry is a fine proxy "
          "and the complementarity framing adds nothing;")
    print("'quality' tracking 'complement' would mean no selection rule is needed "
          "at all — just take the best K.")


if __name__ == "__main__":
    main()
