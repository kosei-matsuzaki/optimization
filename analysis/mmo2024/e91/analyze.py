"""Class ceiling of "uniform restart + local descent" on the GECCO'2024 suite.

Same procedure as entry 88 section 6, applied to the new suite instead of
CEC2013 F14-F20:

  1. draw N uniform starts, descend each with an isotropic CMA-ES (this is
     hunt_coverage.py --null, already run; this script only reads its CSVs),
  2. keep only the descents that actually reached a global minimum
     (best_f <= 1e-5 -- entry 88 section 2: counting nearest-neighbour
     attribution instead flips the sign of the answer),
  3. n = suite budget / mean evaluations per descent = how many restarts one
     competition run can afford,
  4. expected distinct optima of n independent draws from the landing
     distribution = sum_j [1 - (1 - p_j)^n]; divide by K for a PR ceiling.

The ceiling is an upper bound for the whole class: it grants perfect
reporting, no repel bookkeeping and no budget spent on anything but descent.
"""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import niching_by_name          # noqa: E402

EPS = 1e-5
HERE = Path(__file__).resolve().parent


def ceiling(name: str) -> dict:
    with gzip.open(HERE / f"{name}_desc.csv.gz", "rt") as fh:
        rows = list(csv.DictReader(fh))
    b = niching_by_name(name)
    K = int(b.n_global_optima)
    budget = int(b.suite_max_evals)
    evals = np.array([float(r["evals"]) for r in rows])
    best_f = np.array([float(r["best_f"]) for r in rows])
    land = np.array([int(r["land_opt"]) for r in rows])
    hit = best_f <= EPS
    n = budget / evals.mean()
    p = np.zeros(K)
    for j in land[hit]:
        p[j] += 1
    p /= len(rows)                      # failures keep their share of the mass
    exp_distinct = float(np.sum(1.0 - (1.0 - p) ** n))
    return {
        "func": name, "K": K, "draws": len(rows), "hit_rate": float(hit.mean()),
        "mean_evals": float(evals.mean()), "restarts_per_run": n,
        "reachable": int((p > 0).sum()), "exp_distinct": exp_distinct,
        "pr_ceiling": exp_distinct / K,
    }


def main() -> None:
    names = sorted(p.name[: -len("_desc.csv.gz")] for p in HERE.glob("*_desc.csv.gz"))
    out = [ceiling(nm) for nm in names]
    hdr = ("func", "K", "draws", "hit_rate", "mean_evals", "restarts_per_run",
           "reachable", "exp_distinct", "pr_ceiling")
    print(" ".join(f"{h:>16}" for h in hdr))
    for r in out:
        print(" ".join(f"{r[h]:>16.4g}" if isinstance(r[h], float)
                       else f"{str(r[h]):>16}" for h in hdr))
    with open(HERE / "ceiling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)
    print(f"\nmean PR ceiling over {len(out)} problems: "
          f"{np.mean([r['pr_ceiling'] for r in out]):.4f}")


if __name__ == "__main__":
    main()
