#!/usr/bin/env python3
"""B4: does the acceptance set collapse to one component as dimension rises?

Grid flood-fill (scripts/acceptance_components.py) stops being possible past 3-D,
so components are estimated two ways and the two are compared against the grid
truth at dim 2 before being trusted higher up:

  uniform   sample the box, keep points with f <= tau, join two kept points when
            they are close AND the segment between them stays <= tau, count graph
            components. The naive estimator; expected to fall apart in high
            dimension because uniform samples never land close to each other.

  anchor    multistart local search to get local minima, keep those with
            f <= tau, join two anchors when the segment between them stays
            <= tau. Counts wells rather than sample points, so it survives
            dimensions where uniform sampling cannot.

Both over-count: a curved connection that leaves the straight segment is missed,
so a reported K is an upper bound on the true component count. The point of the
dim-2 calibration is to measure how bad that bias is where the truth is known.

Usage:
  python3 scripts/components_by_dim.py [--dims 2,3,5,10] [--quantile 0.01]
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

# One per structural family, so a collapse (or a blow-up) can be attributed.
_FUNCS = [
    "F01-Sphere",            # truth: 1 component at every tau
    "F08-Rosenbrock",        # truth: 1 (bent valley) — tests the thin-set failure
    "F03-RastriginSep",      # separable multimodal: wells multiply with dimension
    "F15-RastriginRot",      # same, rotated (non-separable)
    "F21-Gallagher101",      # fixed number of wells regardless of dimension
    "F20-Schwefel",          # deceptive, wells far apart
]


def _segment_ok(f, a: np.ndarray, b: np.ndarray, tau: float, s: int = 64) -> bool:
    """Does the straight segment a-b stay inside the acceptance set?"""
    for t in np.linspace(0.0, 1.0, s + 2)[1:-1]:
        if f(a + t * (b - a)) > tau:
            return False
    return True


def _components(pts: np.ndarray, f, tau: float, radius: float) -> int:
    """Components of the graph joining points that are near and segment-connected."""
    n = len(pts)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if float(np.linalg.norm(pts[i] - pts[j])) > radius:
                continue
            if _segment_ok(f, pts[i], pts[j], tau):
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def _anchors(f, lo: float, hi: float, dim: int, n_starts: int,
             rng: np.random.Generator, tol: float) -> np.ndarray:
    """Local minima from multistart L-BFGS-B, deduplicated by distance."""
    found: list[np.ndarray] = []
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi, dim)
        res = minimize(f, x0, method="L-BFGS-B", bounds=[(lo, hi)] * dim,
                       options={"maxfun": 2000})
        x = np.clip(res.x, lo, hi)
        if all(float(np.linalg.norm(x - y)) > tol for y in found):
            found.append(x)
    return np.array(found) if found else np.zeros((0, dim))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=str, default="2,3,5,10")
    ap.add_argument("--admit", type=str, default="0.25,0.5,0.75,1.0",
                    help="tau = the p-quantile of the ANCHOR values, i.e. admit the "
                         "best p of the wells found. Neither a quantile of uniform f "
                         "nor a fraction of the median is comparable across functions "
                         "and dimensions (uniform samples degrade with dimension, and "
                         "f ranges differ by orders of magnitude); admitting a fixed "
                         "share of the wells is. It also traces the merge tree "
                         "directly: as more wells are admitted, do they stay separate "
                         "or join?")
    ap.add_argument("--n-uniform", type=int, default=20000)
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dims = [int(s) for s in args.dims.split(",")]
    admits = [float(s) for s in args.admit.split(",")]
    print(f"acceptance components vs dimension   tau = p-quantile of anchor values"
          f"   multistart={args.n_starts}")
    head = f"{'function':<20}{'dim':>4}{'n_anch':>8}" + "".join(
        f"{f'p={a:g}':>14}" for a in admits)
    print(head)
    print("-" * len(head))
    for name in _FUNCS:
        for dim in dims:
            reg = _REGISTRY.get(dim)
            if reg is None or name not in reg:
                continue
            b = reg[name]
            lo, hi = b.bounds
            span = hi - lo
            f = lambda x: float(b.func(np.asarray(x, dtype=float)))  # noqa: E731
            rng = np.random.default_rng(args.seed)

            # Anchors do not depend on tau, so multistart runs once per (f, dim).
            anch_all = _anchors(f, lo, hi, dim, args.n_starts, rng, tol=0.01 * span)
            f_anch = np.array([f(x) for x in anch_all]) if len(anch_all) else np.zeros(0)

            cells = []
            for p in admits:
                if not len(anch_all):
                    cells.append(f"{0:>4}->{0:<3}".rjust(14))
                    continue
                tau = float(np.quantile(f_anch, p))
                sel = anch_all[f_anch <= tau]
                k = _components(sel, f, tau, span * dim ** 0.5) if len(sel) else 0
                cells.append(f"{len(sel):>4}->{k:<3}".rjust(14))
            print(f"{name:<20}{dim:>4}{len(anch_all):>8}" + "".join(cells))
    print("\ncells: anchors below tau -> connected components among them.")
    print("straight-segment tests miss curved connections, so K is an upper bound.")


if __name__ == "__main__":
    main()
