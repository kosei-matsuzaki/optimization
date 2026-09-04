#!/usr/bin/env python3
"""Does the 1/10-budget ranking survive the suite's own budget?

Every method comparison in docs/ so far was taken at --evals-frac 0.1 with 5-30
seeds, because the CEC2013 niching budgets (50k-400k evaluations) were too slow
to iterate on locally. That is the weakest link in status.md: a reviewer asks it
first. `.github/workflows/run.yml` (mode: niching) now runs the same
`niching_baseline.py` cells at --evals-frac 1.0, one job per function. This
reads those per-function CSVs back and reports, at the judgement levels only
(eps <= 1e-3, log entry 28's operating rule):

  1. the per-function ranking by mean peak ratio, and
  2. the paired MC-ESO vs NMMSO contrast per seed -- w/t/l, two-sided Wilcoxon
     signed-rank p, and A12 -- because the pre-registered rejection condition is
     stated as a sign, not as a mean.

Pre-registered rejection condition (docs/research_loop.md, open question 2):
the 1/10-budget ranking is not preserved at full budget -- in particular if
"MC-ESO >= NMMSO on the two Shubert functions" flips, the adoption candidate's
foundation goes with it. REFERENCE_1E3 below is that 1/10-budget table, copied
from status.md before the full-budget numbers existed, so the comparison is
against a written-down expectation and not a remembered one.

NM-Restart's N06 / N08 rows are invalid (log entry 28: restart count is 1
because _XATOL=1e-12 stops the simplex from converging). They are dropped here
and the drop is printed -- silently dropping cells has changed a ranking in this
project before.

``--pair A,B`` swaps the contrasted pair. It defaults to MC-ESO,NMMSO (the
question this script was written for); entry 51 uses ``MC-ESO-rel,MC-ESO`` to
put the adoption candidate against base at the same budget, off CSVs written by
the same driver. The 1/10-budget reference block below only applies to the
NMMSO pair and is printed for that pair only.

Usage:
  python3 scripts/fullbudget_rank.py analysis/hm/fullbudget_*.csv
  python3 scripts/fullbudget_rank.py --pair MC-ESO-rel,MC-ESO analysis/hm/e51/*.csv
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# status.md "ゴールとの距離", PR@1e-3, 1/10 budget, 5 seeds. The sign of
# (MC-ESO - NMMSO) here is what the full-budget run has to reproduce.
REFERENCE_1E3 = {
    "N06-Shubert2D": (0.811, 0.778),
    "N07-Vincent2D": (0.239, 0.994),
    "N08-Shubert3D": (0.121, 0.111),
    "N09-Vincent3D": (0.115, 0.460),
    "N10-ModRastrigin2D": (0.917, 1.000),
}
INVALID = {("NM-Restart", "N06-Shubert2D"), ("NM-Restart", "N08-Shubert3D")}
JUDGEMENT = ("pr_1e-3", "pr_1e-5")


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x == y), the Vargha-Delaney effect size."""
    gt = sum((a > b) for a in x for b in y)
    eq = sum((a == b) for a in x for b in y)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def paired(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int, float, float]:
    w = int((x > y).sum())
    t = int((x == y).sum())
    lo = int((x < y).sum())
    p = 1.0 if w + lo == 0 else wilcoxon(x, y).pvalue
    return w, t, lo, p, a12(x, y)


def main() -> None:
    argv = sys.argv[1:]
    lhs, rhs = "MC-ESO", "NMMSO"
    if "--pair" in argv:
        i = argv.index("--pair")
        lhs, rhs = (s.strip() for s in argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    paths = [Path(p) for p in argv]
    if not paths:
        raise SystemExit(__doc__)

    # rows[func][method][level] -> per-seed peak ratios, ordered by seed
    rows: dict = defaultdict(lambda: defaultdict(dict))
    budgets: dict = {}
    dropped = []
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                fn, m = r["function"], r["method"]
                if (m, fn) in INVALID:
                    dropped.append((m, fn))
                    continue
                budgets[fn] = int(r["evals"])
                for lvl in JUDGEMENT:
                    rows[fn][m].setdefault(lvl, []).append(float(r[lvl]))

    if dropped:
        n = len(dropped)
        cells = sorted({f"{m} x {fn}" for m, fn in dropped})
        print(f"dropped {n} invalid rows (log entry 28): {', '.join(cells)}\n")

    for fn in sorted(rows):
        methods = rows[fn]
        seeds = len(next(iter(methods.values()))["pr_1e-3"])
        print(f"=== {fn}   budget {budgets[fn]}   {seeds} seeds")
        print(f"{'method':<14}" + "".join(f"{lvl:>10}" for lvl in JUDGEMENT))
        order = sorted(methods, key=lambda m: -np.mean(methods[m]["pr_1e-3"]))
        for m in order:
            cells = "".join(f"{np.mean(methods[m][lvl]):>10.3f}"
                            for lvl in JUDGEMENT)
            print(f"{m:<14}{cells}")

        if lhs in methods and rhs in methods:
            print(f"  {f'{lhs} vs {rhs}':<20}{'w/t/l':>10}{'p':>10}{'A12':>8}")
            for lvl in JUDGEMENT:
                x = np.array(methods[lhs][lvl])
                y = np.array(methods[rhs][lvl])
                w, t, lo, p, a = paired(x, y)
                print(f"  {lvl:<20}{f'{w}/{t}/{lo}':>10}{p:>10.4f}{a:>8.2f}")
            ref = REFERENCE_1E3.get(fn) if (lhs, rhs) == ("MC-ESO", "NMMSO") else None
            if ref:
                exp = "MC-ESO>=NMMSO" if ref[0] >= ref[1] else "MC-ESO<NMMSO"
                got_v = (np.mean(methods["MC-ESO"]["pr_1e-3"])
                         >= np.mean(methods["NMMSO"]["pr_1e-3"]))
                got = "MC-ESO>=NMMSO" if got_v else "MC-ESO<NMMSO"
                mark = "held" if exp == got else "FLIPPED"
                print(f"  1/10 budget said {exp} (PR@1e-3 "
                      f"{ref[0]:.3f} vs {ref[1]:.3f}); full budget says {got}"
                      f"  -> {mark}")
        print()


if __name__ == "__main__":
    main()
