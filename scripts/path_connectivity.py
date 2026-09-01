#!/usr/bin/env python3
"""C2: how often does the straight-segment test miss a real connection?

Every component count in this project so far joins two wells only when the
straight segment between them stays inside the acceptance set. That test can
only err one way — it misses connections that need a curved path — so reported
component counts are upper bounds. B4 concluded that acceptance sets do not
collapse in higher dimension, but a straight-segment test also fails more often
in higher dimension, so the conclusion and the artefact point the same way and
cannot be told apart from that experiment alone.

This measures the artefact directly. For pairs of anchors the straight test
calls disconnected, it searches for a bent path:

  bend1   one free midpoint, chosen to minimise the highest f along the two
          segments (Nelder-Mead over the midpoint)
  bend2   two free midpoints, three segments

A pair rescued by a bent path was a false split. The rescue rate is the
over-count bias of the straight test, per function and dimension.

Usage:
  python3 scripts/path_connectivity.py [--dims 2,5,10] [--admit 0.5] [--n-starts 60]
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import (BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME,   # noqa: E402
                             BENCHMARKS_5D_BY_NAME, BENCHMARKS_10D_BY_NAME)

_REGISTRY = {2: BENCHMARKS_BY_NAME, 3: BENCHMARKS_3D_BY_NAME,
             5: BENCHMARKS_5D_BY_NAME, 10: BENCHMARKS_10D_BY_NAME}
_FUNCS = ["F08-Rosenbrock", "F15-RastriginRot", "F21-Gallagher101", "F20-Schwefel"]


def _path_max(f, pts: list[np.ndarray], s: int) -> float:
    """Highest f along a polyline through pts (endpoints included)."""
    worst = -np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, s):
            worst = max(worst, f(a + t * (b - a)))
    return worst


def _bent_ok(f, a: np.ndarray, b: np.ndarray, tau: float, n_mid: int,
             s: int, rng: np.random.Generator) -> bool:
    """Can a polyline with n_mid free midpoints stay inside the acceptance set?

    The midpoints start on the straight segment and are then moved to minimise
    the highest f along the path — a crude string method, which is enough to
    tell "a bent path exists" from "there is a real barrier".
    """
    dim = len(a)
    x0 = np.concatenate([a + (i + 1) / (n_mid + 1) * (b - a) for i in range(n_mid)])

    def obj(z):
        mids = [z[i * dim:(i + 1) * dim] for i in range(n_mid)]
        return _path_max(f, [a, *mids, b], s)

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxfev": 300 * n_mid * dim, "xatol": 1e-3, "fatol": 1e-3})
    return float(res.fun) <= tau


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=str, default="2,5,10")
    ap.add_argument("--admit", type=float, default=0.5)
    ap.add_argument("--n-starts", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"straight-vs-bent connectivity   admit={args.admit}  "
          f"multistart={args.n_starts}  pairs<={args.max_pairs}  seg-samples={args.samples}")
    print(f"{'function':<20}{'dim':>4}{'anchors':>9}{'split pairs':>13}"
          f"{'bend1 ok':>10}{'bend2 ok':>10}{'rescued':>9}")
    print("-" * 75)
    for name in _FUNCS:
        for dim in [int(s) for s in args.dims.split(",")]:
            reg = _REGISTRY.get(dim)
            if reg is None or name not in reg:
                continue
            b = reg[name]
            lo, hi = b.bounds
            f = lambda x: float(b.func(np.clip(np.asarray(x, float), lo, hi)))  # noqa: E731
            rng = np.random.default_rng(args.seed)

            anchors: list[np.ndarray] = []
            for _ in range(args.n_starts):
                r = minimize(f, rng.uniform(lo, hi, dim), method="L-BFGS-B",
                             bounds=[(lo, hi)] * dim, options={"maxfun": 2000})
                x = np.clip(r.x, lo, hi)
                if all(float(np.linalg.norm(x - y)) > 0.01 * (hi - lo) for y in anchors):
                    anchors.append(x)
            if len(anchors) < 2:
                print(f"{name:<20}{dim:>4}{len(anchors):>9}{'-':>13}{'-':>10}{'-':>10}{'-':>9}")
                continue
            fa = np.array([f(x) for x in anchors])
            tau = float(np.quantile(fa, args.admit))
            keep = [x for x, v in zip(anchors, fa) if v <= tau]

            split = [(i, j) for i, j in itertools.combinations(range(len(keep)), 2)
                     if _path_max(f, [keep[i], keep[j]], args.samples) > tau]
            rng.shuffle(split)
            split = split[:args.max_pairs]
            n1 = sum(_bent_ok(f, keep[i], keep[j], tau, 1, args.samples, rng)
                     for i, j in split)
            rest = [(i, j) for (i, j) in split
                    if not _bent_ok(f, keep[i], keep[j], tau, 1, args.samples, rng)]
            n2 = sum(_bent_ok(f, keep[i], keep[j], tau, 2, args.samples, rng)
                     for i, j in rest)
            rate = (n1 + n2) / len(split) if split else float("nan")
            print(f"{name:<20}{dim:>4}{len(keep):>9}{len(split):>13}"
                  f"{n1:>10}{n2:>10}{rate:>8.0%}")
    print("\nrescued = pairs the straight test called disconnected that a bent path "
          "connects; that fraction of the reported components was spurious.")


if __name__ == "__main__":
    main()
