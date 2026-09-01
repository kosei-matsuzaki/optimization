#!/usr/bin/env python3
"""B1 premise test: how many connected components does an acceptance set have?

The proposed formulation ("distinct solutions = connected components of
{x : f(x) <= tau}") is only worth pursuing if real acceptance sets actually
split into several components at thresholds a user would choose. Nguyen (2019)
proved every sublevel set of a neural-network loss is connected, so the premise
cannot be assumed — it has to be measured.

This computes the ground truth in 2-D by flood-filling a dense grid, for every
benchmark in the project's registries, at thresholds set as quantiles of f over
the grid (a user picks tau by "how good is good enough", which is what a
quantile expresses without needing to know f_opt).

Reported per (function, tau):
  K        number of connected components (4-neighbour flood fill)
  K_big    components holding at least `--min-frac` of the accepted cells —
           single-cell specks are grid artefacts, not design alternatives
  area     fraction of the domain that is accepted
  r_max    largest inscribed radius over components, in units of the domain
           span — the tolerance the best component offers

Usage:
  python3 scripts/acceptance_components.py [--grid 600] [--suite bbob|niching]
"""
from __future__ import annotations
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import (BENCHMARKS_BY_NAME,                 # noqa: E402
                             NICHING_BENCHMARKS_BY_NAME)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-neighbour connected components of a boolean grid, as index arrays."""
    n, m = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for i0 in range(n):
        for j0 in range(m):
            if not mask[i0, j0] or seen[i0, j0]:
                continue
            q = deque([(i0, j0)])
            seen[i0, j0] = True
            cells = []
            while q:
                i, j = q.popleft()
                cells.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < n and 0 <= b < m and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        q.append((a, b))
            out.append(np.array(cells))
    return out


def _inscribed_radius(mask: np.ndarray, cells: np.ndarray, step: float) -> float:
    """Largest disc inside this component, approximated on the grid.

    Chebyshev-style erosion: repeatedly peel the boundary and count how many
    peels the component survives. Cheap and good enough to rank components by
    how much room they give.
    """
    sub = np.zeros_like(mask)
    sub[cells[:, 0], cells[:, 1]] = True
    peels = 0
    while sub.any():
        nxt = (sub
               & np.roll(sub, 1, 0) & np.roll(sub, -1, 0)
               & np.roll(sub, 1, 1) & np.roll(sub, -1, 1))
        nxt[0, :] = nxt[-1, :] = nxt[:, 0] = nxt[:, -1] = False
        if not nxt.any():
            break
        sub = nxt
        peels += 1
    return peels * step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=600)
    ap.add_argument("--suite", choices=["bbob", "niching"], default="niching")
    ap.add_argument("--quantiles", type=str, default="0.001,0.01,0.05,0.10")
    ap.add_argument("--min-frac", type=float, default=0.01,
                    help="a component counts as real if it holds this fraction "
                         "of the accepted cells")
    args = ap.parse_args()

    qs = [float(s) for s in args.quantiles.split(",")]
    if args.suite == "niching":
        benches = [b for b in NICHING_BENCHMARKS_BY_NAME.values() if b.dim == 2]
    else:
        benches = [b for n, b in sorted(BENCHMARKS_BY_NAME.items())
                   if n.startswith("F") and b.dim == 2]

    print(f"acceptance-set components   suite={args.suite}  grid={args.grid}^2  "
          f"quantiles={qs}  min_frac={args.min_frac}")
    header = "function".ljust(22) + "".join(f"{f'q={q:g}':>18}" for q in qs)
    print(header)
    print("-" * len(header))
    for b in benches:
        lo, hi = b.bounds
        xs = np.linspace(lo, hi, args.grid)
        step = (hi - lo) / (args.grid - 1) / (hi - lo)      # span-relative
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        F = np.empty_like(X)
        for i in range(args.grid):
            for j in range(args.grid):
                F[i, j] = b.func(np.array([X[i, j], Y[i, j]]))
        cells = []
        for q in qs:
            tau = float(np.quantile(F, q))
            mask = F <= tau
            comps = _components(mask)
            total = sum(len(c) for c in comps)
            big = [c for c in comps if len(c) >= args.min_frac * total]
            r = max((_inscribed_radius(mask, c, step) for c in big), default=0.0)
            cells.append(f"{len(comps):>4}/{len(big):<3} r={r:.3f}".rjust(18))
        print(b.name.ljust(22) + "".join(cells))
    print("\ncolumns: K_all/K_big  r=largest inscribed radius (span-relative)")


if __name__ == "__main__":
    main()
