#!/usr/bin/env python3
"""How many distinct niches does an f-only top-N trim throw away?

Two caps in this pipeline keep only the N best points *by objective value* and
are blind to which niche a point sits in:

  the scorer's cap   max(100, 2K) points, applied to what the run reports
                     (CEC2013 rule; measured externally in e57)
  MC-ESO's own cap   ``solution_archive_max = 200`` (``mceso.py:822``), applied
                     to the answer archive — one point per abandoned basin,
                     trimmed with ``np.argsort(sol_archive_f)[:200]``

Both are the same operation, so both are measured the same way here. Because
keeping the best N seen so far at every overflow is identical to keeping the
global best N, the offline trim below reproduces the online one exactly (up to
tie-breaking), and no re-optimisation is needed: the hunt dump from
``diagnose_niching.py --hunt-csv`` already holds every endpoint with its f.

The comparison is strictly paired — the same run scored two ways, the design
rule from e28 — so the loss is attributable to the trim and nothing else.

Usage:  python3 scripts/trim_loss.py --cap 200 analysis/hm/e58/n09_full_hunts_*.csv.gz
"""
from __future__ import annotations
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = (1e-1, 1e-3, 1e-5)


def _open(path: str):
    return (gzip.open(path, "rt", newline="") if path.endswith(".gz")
            else open(path, newline=""))


def _rho_K(name: str) -> tuple[float, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME  # noqa: E402
    b = NICHING_BENCHMARKS_BY_NAME[name]
    return float(b.niche_rho), int(b.n_global_optima)


def _greedy(pts: list[tuple[float, np.ndarray]], rho: float, eps: float) -> int:
    """CEC2013's rule: best-f first, accept a point only if it is deeper than
    eps and further than rho from every point already accepted."""
    held: list[np.ndarray] = []
    for f, x in sorted(pts, key=lambda p: p[0]):
        if f > eps:
            break
        if not held or min(float(np.linalg.norm(y - x)) for y in held) > rho:
            held.append(x)
    return len(held)


def main() -> None:
    args = sys.argv[1:]
    cap = 200
    if "--cap" in args:
        i = args.index("--cap")
        cap = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__)
        raise SystemExit(2)

    runs: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = defaultdict(list)
    for p in paths:
        with _open(p) as fh:
            for r in csv.DictReader(fh):
                xs = [float(r[k]) for k in r
                      if k.startswith("x") and k[1:].isdigit()]
                runs[(r["function"], int(r["seed"]))].append(
                    (float(r["f"]), np.array(xs)))

    for name in sorted({n for n, _ in runs}):
        rho, K = _rho_K(name)
        seeds = sorted(s for n, s in runs if n == name)
        print(f"\n=== {name}  K={K}  rho={rho}  cap={cap}  "
              f"({len(seeds)} seeds) ===")
        print("Distinct niches among the hunt endpoints, before and after an "
              "f-only top-N trim.")
        for eps in LEVELS:
            rows = []
            for s in seeds:
                pts = runs[(name, s)]
                kept = sorted(pts, key=lambda p: p[0])[:cap]
                rows.append((len(pts), _greedy(pts, rho, eps),
                             _greedy(kept, rho, eps)))
            a = np.array(rows, dtype=float)
            print(f"\n  eps = {eps:g}")
            print(f"    {'seed':>5}{'endpoints':>11}{'niches all':>12}"
                  f"{'niches trimmed':>16}{'lost':>7}")
            for s, (n, full, trim) in zip(seeds, rows):
                print(f"    {s:>5}{n:>11d}{full:>12d}{trim:>16d}"
                      f"{full - trim:>7d}")
            print(f"    {'mean':>5}{a[:, 0].mean():>11.1f}{a[:, 1].mean():>12.2f}"
                  f"{a[:, 2].mean():>16.2f}{(a[:, 1] - a[:, 2]).mean():>7.2f}")
            worse = int((a[:, 2] < a[:, 1]).sum())
            over = int((a[:, 0] > cap).sum())
            try:
                from scipy.stats import wilcoxon
                p = float(wilcoxon(a[:, 1], a[:, 2]).pvalue) \
                    if worse else float("nan")
            except Exception:
                p = float("nan")
            print(f"    endpoints over the cap in {over}/{len(seeds)} runs; "
                  f"trim loses niches in {worse}/{len(seeds)} "
                  f"(paired Wilcoxon p = {p:.2g})")
            print(f"    PR from all endpoints {a[:, 1].mean() / K:.3f} "
                  f"vs from the trimmed set {a[:, 2].mean() / K:.3f} of K = {K}")


if __name__ == "__main__":
    main()
