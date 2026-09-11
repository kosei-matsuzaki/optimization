"""Is the chained descent in RestartLanderOptimizer the same descent the saved
offline dumps were drawn with?

The run and the estimate must differ in exactly the three ways entry 110 is
about (restart count, the report cap, one run's landings vs an i.i.d. draw from
500).  They must not differ in the descent itself.  This forces the optimizer's
draw k to be `_null_descent`'s draw k -- same x0, same CMA seed -- and checks
best_f / evals / landing against `analysis/mmo2024/e98/M09-D10-PIN01_sig100200`
(sigma0 = 0.1 x span, per-descent cap 12500), which is the dump family entry
103's 0.5529 was computed from.

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e110/identity_check.py
"""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name                        # noqa: E402
from core.optimizers.restart_lander import RestartLanderOptimizer  # noqa: E402

NAME = "M09-D10-PIN01"
DUMP = ROOT / "analysis/mmo2024/e98" / f"{NAME}_sig100200.csv.gz"
NDRAW = 4


class _FixedDraws:
    """Hands out `_null_descent`'s draw k, in order, in place of the run's rng."""

    def __init__(self, first):
        self.k = first

    def uniform(self, lo, hi, size):
        x = np.random.default_rng(1_000_000 + self.k).uniform(lo, hi, size=size)
        self.k += 1
        return x


def main() -> None:
    rows = list(csv.DictReader(gzip.open(DUMP, "rt")))
    want = {int(r["draw"]): r for r in rows}
    b = niching_by_name(NAME)
    opts = np.asarray(b.optima_pos, dtype=float)
    # seed 0 => the optimizer's CMA seed for descent k is k+1, which is what
    # _null_descent passes.  Budget = NDRAW full descents' worth, so no descent
    # is cut by the run budget.
    opt = RestartLanderOptimizer(b, seed=0, sigma_ratio=0.1, descent_budget=12500)
    opt.rng_override = _FixedDraws(0)
    opt.optimize(12500 * NDRAW)
    ok = True
    print(f"{'draw':>5}{'evals':>8}{'evals*':>8}{'best_f':>14}{'best_f*':>14}"
          f"{'land':>6}{'land*':>6}  stop")
    for r in opt.descents[:NDRAW]:
        k = r["descent"]
        w = want[k]
        j = int(np.argmin(np.linalg.norm(opts - r["x"], axis=1)))
        same = (r["evals"] == int(float(w["evals"]))
                and abs(r["best_f"] - float(w["best_f"])) <= 1e-12 * max(
                    1.0, abs(float(w["best_f"])))
                and j == int(w["land_opt"]))
        ok &= same
        print(f"{k:>5}{r['evals']:>8}{int(float(w['evals'])):>8}"
              f"{r['best_f']:>14.6g}{float(w['best_f']):>14.6g}"
              f"{j:>6}{int(w['land_opt']):>6}  {r['stop']}"
              f"{'' if same else '   <-- MISMATCH'}")
    print("identity check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
